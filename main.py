import torch 
import torch.nn as nn 
import torch.nn.functional as F
from .extractor.ResNet12 import ResNet12 
from torchvision import models, datasets, transform
from torch.utils.data import DataLoader

#pretrained feature extractor(resnet12)

#feature_extractor = ResNet12(args)
#The feature extractor part is kept frozen for whole retraining process
#I can just use pretrained resnet18
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
        features = self.feature_extractor(x).squeeze()
        hv_features = self.fc(features)
        return hv_features

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


# CIFAR-100 transfrom
transform = transforms.Compose([
    transforms.Resize((84, 84)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # Normalizing the dataset
])

# Load CIFAR-100 train and test datasets
train_data = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
test_data = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)

# Create data loaders
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)
