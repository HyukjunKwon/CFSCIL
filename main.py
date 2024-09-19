import torch 
import torch.nn as nn 
import torch.nn.functional as F
from .extractor.ResNet12 import ResNet12 
from torchvision import models

#pretrained feature extractor(resnet12)

#feature_extractor = ResNet12(args)
#The feature extractor part is kept frozen for whole retraining process
#I can just use pretrained resnet18
class FEwHD(nn.Module):
    def __init__(self, feature_dim = 512, hv_dim = 3000):
        super(FEwHD, self).__init__()
        resnet18 = models.resnet18(pretrained = True)
        feature_extractor = nn.Sequential(*list(resnet18.children())[:-1]) #cutting last layer
        for param in resnet18.parameters():
            param.requires_grad = False
        #FC memory for HD mapping

        self.fc = nn.Linear(feature_dim, hv_dim)
    def foward(self, x):
        features = self.feature_extractor(x).squeeze()

        hv_features = self.fc(features)
        return hv_features
#Class memory

class CFSCIL_Mem:
    def __init__(self):
        self.memory = {}

    def update_memory(self, class_id, features):
        self.memory[class_id] = features.mean(dim=0)

    def classify(self, input_features):
        similarities = {
            class_id: F.cosine_similarity(input_features, class_proto, dim=0).item()
            for class_id, class_proto in self.memory.items()
        }
        return max(similarities, key=similarities.get)  # Class ID with highest similarity



