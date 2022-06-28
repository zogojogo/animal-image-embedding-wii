import torch.nn as nn
from torchvision import models

class FeatureEmbedding(nn.Module):
  def __init__(self):
    super().__init__()
    model = models.efficientnet_b0(pretrained=True)
    model = model.to('cuda')
    model.eval()
    self.feature_embedding = nn.Sequential(*(list(model.children())[:-1]))

  def forward(self, x):
    x = self.feature_embedding(x)
    return x[0, :, 0, 0]