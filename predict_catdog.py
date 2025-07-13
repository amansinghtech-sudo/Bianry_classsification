import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image


class CatDogCNN(nn.Module):
    def __init__(self):
        super(CatDogCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 128), nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CatDogCNN().to(device)
model.load_state_dict(torch.load('/home/aman/Documents/modle/catdog_cnn.pth', map_location=device))
model.eval()


transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

image_path = "/home/aman/Documents/modle/test.jpg"  

try:
    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)  # Add batch dim

    with torch.no_grad():
        output = model(img_tensor)
        _, predicted = torch.max(output, 1)

    class_names = ['Dog', 'Cat']
    print(f" Image: {image_path}")
    print(f" Prediction: {class_names[predicted.item()]}")
except Exception as e:
    print(f" Error: {e}")
