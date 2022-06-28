import torch
from torchvision import transforms, models
from PIL import Image
import io
import timm

class Img2VecConvNext():
    def __init__(self, mode='initialize'):
        self.device = torch.device("cuda")
        self.model_path = 'models/effnet.pth'
        self.model, self.featureLayer = self.getFeatureLayer() if mode == 'initialize' else self.load_model(self.model_path)
        self.numberFeatures = 1280
        self.model = self.model.to(self.device)
        self.model.eval()
        self.toTensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def load_model(self, model_path):
        model = torch.load(model_path)
        layer = model._modules.get('avgpool')
        return model, layer

    def getFeatureLayer(self):
        cnnModel = models.efficientnet_b0(pretrained=True)
        layer = cnnModel._modules.get('avgpool')
        return cnnModel, layer

    def getVec(self, path, mode):
        if mode == 'api':
            img = Image.open(io.BytesIO(path))
        else:
            img = Image.open(path)
        image = self.normalize(self.toTensor(img)).unsqueeze(0).to(self.device)
        embedding = torch.zeros(1, self.numberFeatures, 1, 1)
        def copyData(m, i, o): embedding.copy_(o.data)
        h = self.featureLayer.register_forward_hook(copyData)
        self.model(image)
        h.remove()
        return embedding.numpy()[0, :, 0, 0]