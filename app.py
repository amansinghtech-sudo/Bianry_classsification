import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import io

# Define the same model structure
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

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CatDogCNN().to(device)
model.load_state_dict(torch.load('catdog_cnn.pth', map_location=device))
model.eval()

# Define image transformation
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# Streamlit UI
st.title("Cat vs Dog Classifier 🐱🐶")
st.write("Upload an image of a **cat** or a **dog**, and the model will predict which one it is!")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess and predict
    input_tensor = transform(image).unsqueeze(0).to(device)
    output = model(input_tensor)
    _, predicted = torch.max(output, 1)
    class_names = ['Dog', 'Cat']
    prediction = class_names[predicted.item()]

    st.write(f"### Prediction: **{prediction}**")
