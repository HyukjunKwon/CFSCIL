import torch 
import torch.nn as nn 
import torch.nn.functional as F
#from .extractor.ResNet12 import ResNet12 
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm 

#pretrained feature extractor(resnet12)

#feature_extractor = ResNet12(args)
#The feature extractor part is kept frozen for whole retraining process
#I can just use pretrained resnet18
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FEwHD(nn.Module):
    def __init__(self, feature_dim = 512, hv_dim = 3000):
        super(FEwHD, self).__init__()
        resnet18 = models.resnet18(pretrained = True)
        self.feature_extractor = nn.Sequential(*list(resnet18.children())[:-1]) #cutting last layer
        for param in resnet18.parameters():
            param.requires_grad = False
        
        #FC memory for HD mapping
        self.fc = nn.Linear(feature_dim, hv_dim)
    def forward(self, x):
        x = x.to(device)
        features = self.feature_extractor(x).squeeze()
        hv_features = self.fc(features)
        return hv_features

def orthogonality_loss(fc_weights, epsilon=1e-5):
    # Normalize the weight vectors (class hypervectors)
    fc_weights_normalized = F.normalize(fc_weights, p=2, dim=1)  # Normalize along the feature dimension

    # Compute the dot product between all pairs of weight vectors
    dot_products = torch.matmul(fc_weights_normalized, fc_weights_normalized.T)

    # Subtract identity matrix to ensure only off-diagonal terms are considered (to avoid self dot products)
    identity_matrix = torch.eye(dot_products.size(0)).to(fc_weights.device)
    orth_loss = torch.sum(torch.abs(dot_products - identity_matrix))

    return orth_loss

#Class memory

class CFSCILMemory:
    def __init__(self):
        self.memory = {}  # Store class prototypes

    # Update memory with nudging for previously stored prototypes
    def update_memory_with_nudging(self, class_id, features, learning_rate=0.1):
        if class_id in self.memory:
            # Nudging
            self.memory[class_id] = (1 - learning_rate) * self.memory[class_id] + learning_rate * features.mean(dim=0)
        else:
            # For new classes, simply add the prototype
            self.memory[class_id] = features.mean(dim=0)

    # Classify using cosine similarity to class prototypes
    def classify(self, input_features):
        similarities = {
            class_id: F.cosine_similarity(input_features, class_proto, dim=0).item()
            for class_id, class_proto in self.memory.items()
        }
        return max(similarities, key=similarities.get)  # Class ID with highest similarity


# Function to get CIFAR-100 dataset loaders
def get_cifar100_loaders(batch_size=64):
    transform = transforms.Compose([
        transforms.Resize((84, 84)),  # Resizing to match miniImageNet input size
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # Normalizing the dataset
    ])

    # Load CIFAR-100 train and test datasets
    train_data = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
    test_data = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)

    # Create data loaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

def train(model, memory, optimizer, train_loader, num_epochs=10, orth_weight=0.001):
    model.train()

    for epoch in range(num_epochs):
        correct, total = 0, 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for data, labels in progress_bar:
            optimizer.zero_grad()

            # Move data and labels to the correct device
            data, labels = data.to(device), labels.to(device)

            # Extract hyperdimensional features from the data
            features = model(data)

            # Update memory with nudging for each class in the batch
            for class_id in torch.unique(labels):
                class_features = features[labels == class_id]
                memory.update_memory_with_nudging(class_id.item(), class_features)

            # Orthogonality loss on the FC layer weights
            orth_loss = orthogonality_loss(model.fc.weight)

            # Combine orthogonality loss and classification loss if memory is not empty
            if memory.memory:
                predictions = [memory.classify(feature) for feature in features]
                correct += (torch.tensor(predictions).to(device) == labels).sum().item()
                total += labels.size(0)

                # Compute combined loss
                total_loss = orth_weight * orth_loss  # Optionally add classification loss if using one
                total_loss.backward()

                # Update tqdm progress bar with accuracy
                progress_bar.set_postfix(accuracy=f"{(correct/total) * 100:.2f}%")
            else:
                # Skip classification accuracy computation if memory is empty
                progress_bar.set_postfix(accuracy="N/A (Memory empty)")

            # Backpropagate and update weights
            optimizer.step()

        # Print accuracy and orthogonality loss for the current epoch if applicable
        if total > 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Training Accuracy: {(correct/total) * 100:.2f}%")
        else:
            print(f"Epoch {epoch+1}/{num_epochs}, No classification yet (Memory still empty)")


# Evaluation loop with CFSCILMemory
def evaluate(model, memory, test_loader):
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for data, labels in test_loader:

            data, labels = data.to(device), labels.to(device)

            # Extract hyperdimensional features
            features = model(data)

            # Classify each sample by comparing to stored prototypes
            predictions = [memory.classify(feature) for feature in features]

            # Compute accuracy
            correct += (torch.tensor(predictions).to(device) == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    print(f"Test Accuracy: {accuracy * 100:.2f}%")


# Main function to wrap data loading, training, and evaluation
def main():
    # Load the dataset
    train_loader, test_loader = get_cifar100_loaders(batch_size=64)

    # Initialize the model, memory, optimizer, and loss function
    model = FEwHD(feature_dim=512, hv_dim=512).to(device)
    memory = CFSCILMemory()
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)
    
    # Train the model
    train(model, memory, optimizer, train_loader, num_epochs=1)

    # Evaluate the model
    evaluate(model, memory, test_loader)


# Run the main function
if __name__ == "__main__":
    main()
