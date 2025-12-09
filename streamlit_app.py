import streamlit as st
import torch
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from models.generator import UNetGenerator

# =====================
# ⚙️ Load Model
# =====================
@st.cache_resource
def load_model():
    model = UNetGenerator()
    model.load_state_dict(torch.load("saved_models/generator_fast.pth", map_location="cpu"))
    model.eval()
    return model

generator = load_model()

# =====================
# 🧠 Streamlit UI
# =====================
st.title("👵 AI-Based Human Face Aging & Rejuvenation ")

uploaded_img = st.file_uploader("📤 Upload a face image (jpg/png)", type=["jpg", "png"])
target_age = st.slider("🎯 Select Target Age", 0, 100, 60)

if uploaded_img:
    image = Image.open(uploaded_img).convert("RGB")
    st.image(image, caption="🧍 Original Face", width=300)

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    input_tensor = transform(image).unsqueeze(0)
    target_age_tensor = torch.tensor([target_age / 100.0]).float().unsqueeze(0)

    with torch.no_grad():
        fake_face = generator(input_tensor, target_age_tensor)

    fake_face = (fake_face.squeeze(0) + 1) / 2
    fake_face = TF.adjust_sharpness(fake_face, 2.0)
    fake_face = transforms.ToPILImage()(fake_face)

    st.image(fake_face, caption=f"👴 Aged Face ({target_age} yrs)", width=300)
    st.success("✅ Transformation Complete — Realistic aging applied!")
