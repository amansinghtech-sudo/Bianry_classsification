import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import torch.nn.functional as F

# Image path
image_path = '/home/aman/Documents/modle/test.jpg'

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Load and preprocess the image
img = Image.open(image_path).convert("RGB")
img_tensor = transform(img).unsqueeze(0).to(device)

# Load ResNet18 model with  final layer
model = models.resnet18()
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load('/home/aman/Documents/modle/resnet18_catdog.pth', map_location=device))
model = model.to(device)
model.eval()

# Predict
with torch.no_grad():
    output = model(img_tensor)
    probs = F.softmax(output, dim=1)
    conf, pred = torch.max(probs, 1)

# Class names 
class_names = ['cat', 'dog']  # Only correct if training folder was sorted alphabetically

print(f"📷 Image: {image_path}")
print(f"🔍 Prediction: {class_names[pred.item()]}")
print(f"✅ Confidence: {conf.item():.4f}")
