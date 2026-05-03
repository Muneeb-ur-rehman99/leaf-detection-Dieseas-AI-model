import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# ---------------- DEVICE ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- CLASS NAMES (PUT YOUR REAL NAMES HERE) ----------------
class_names = [
    "Bacterial_spot",
    "Early_blight",
    "Late_blight",
    "Leaf_Mold",
    "Septoria_leaf_spot",
    "Spider_mites",
    "Target_Spot",
    "Yellow_Leaf_Curl_Virus",
    "Mosaic_virus",
    "Healthy",
    "Other"
]

# ---------------- LOAD MODEL ----------------
model = models.efficientnet_b0(weights=None)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(class_names))

model.load_state_dict(torch.load("best_leaf_model.pth", map_location=device))
model = model.to(device)
model.eval()

# ---------------- IMAGE TRANSFORM ----------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ---------------- STREAMLIT UI ----------------
st.title("🌿 Plant Leaf Disease Detector")
st.write("Upload a leaf image and get prediction")

uploaded_file = st.file_uploader("Choose image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img)
        probs = torch.softmax(output, dim=1)
        confidence, pred = torch.max(probs, 1)

    st.success(f"Prediction: {class_names[pred.item()]}")
    st.info(f"Confidence: {confidence.item()*100:.2f}%")