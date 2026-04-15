import streamlit as st
import torch
import torchxrayvision as xrv
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# Streamlit Page Config
# -------------------------------
st.set_page_config(
    page_title="AI X-ray Diagnosis Support",
    page_icon="🩻",
    layout="centered"
)

st.title("🩻 AI X-ray Diagnosis Support Tool")
st.write("Upload a Chest X-ray + Enter symptoms to get AI-supported findings & diagnosis.")
st.warning("⚠️ Educational tool only. Not a replacement for doctor/radiologist opinion.")

# -------------------------------
# Load Pretrained Model
# -------------------------------
@st.cache_resource
def load_model():
    model = xrv.models.DenseNet(weights="densenet121-res224-all")
    model.eval()
    return model

model = load_model()

# -------------------------------
# Image Preprocessing Function
# -------------------------------
def preprocess_image(img):
    img = img.convert("RGB")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    img_tensor = transform(img)

    # Convert RGB to grayscale approximation (Chest X-rays are grayscale)
    img_tensor = img_tensor.mean(dim=0, keepdim=True)

    return img_tensor.unsqueeze(0)  # shape: [1, 1, 224, 224]

# -------------------------------
# Prediction Function
# -------------------------------
def predict(img_tensor):
    with torch.no_grad():
        preds = model(img_tensor)
        preds = torch.sigmoid(preds).cpu().numpy()[0]
    return preds

# -------------------------------
# Symptom + Findings Reasoning
# -------------------------------
def generate_possible_diagnosis(symptoms, findings_dict):
    diagnosis = []
    differentials = []

    # Simple logic rules (basic clinical reasoning)
    if findings_dict.get("Pneumonia", 0) > 0.5 and "fever" in symptoms.lower():
        diagnosis.append("Community Acquired Pneumonia (CAP)")
        differentials.extend(["Tuberculosis", "Lung abscess", "Aspiration pneumonia"])

    if findings_dict.get("Edema", 0) > 0.5 or findings_dict.get("Cardiomegaly", 0) > 0.5:
        diagnosis.append("Congestive Heart Failure (Pulmonary Edema)")
        differentials.extend(["Renal failure fluid overload", "ARDS"])

    if findings_dict.get("Pneumothorax", 0) > 0.5:
        diagnosis.append("Pneumothorax")
        differentials.extend(["Bullous lung disease", "Traumatic pneumothorax"])

    if findings_dict.get("Pleural Effusion", 0) > 0.5:
        diagnosis.append("Pleural Effusion")
        differentials.extend(["Tuberculosis", "Malignancy", "Parapneumonic effusion"])

    if findings_dict.get("Atelectasis", 0) > 0.5:
        diagnosis.append("Atelectasis / Collapse")
        differentials.extend(["Mucus plugging", "Endobronchial tumor"])

    if len(diagnosis) == 0:
        diagnosis.append("No strong abnormality detected (may be normal)")
        differentials.extend(["Early infection", "Mild asthma", "Non-radiological causes"])

    return diagnosis, list(set(differentials))

# -------------------------------
# UI Inputs
# -------------------------------
symptoms = st.text_area(
    "📝 Enter symptoms (example: fever, cough, chest pain, breathlessness)",
    height=100
)

uploaded_file = st.file_uploader("📤 Upload Chest X-ray (JPG/PNG)", type=["jpg", "jpeg", "png"])

# -------------------------------
# Main Processing
# -------------------------------
if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded X-ray", use_column_width=True)

    st.write("⏳ Processing X-ray...")

    img_tensor = preprocess_image(image)
    preds = predict(img_tensor)

    # Labels from model
    labels = model.pathologies

    findings = dict(zip(labels, preds))

    # Display top findings
    st.subheader("📌 AI Detected Findings (Top 8)")
    sorted_findings = sorted(findings.items(), key=lambda x: x[1], reverse=True)

    for name, prob in sorted_findings[:8]:
        st.write(f"**{name}** : {prob:.2f}")

    # Convert model pathology names into common names
    mapped_findings = {
        "Pneumonia": findings.get("Pneumonia", 0),
        "Cardiomegaly": findings.get("Cardiomegaly", 0),
        "Edema": findings.get("Edema", 0),
        "Pneumothorax": findings.get("Pneumothorax", 0),
        "Pleural Effusion": findings.get("Effusion", 0),
        "Atelectasis": findings.get("Atelectasis", 0),
    }

    # Diagnosis generation
    st.subheader("🩺 Probable Diagnosis Suggestion")
    dx, ddx = generate_possible_diagnosis(symptoms, mapped_findings)

    st.write("### ✅ Most likely diagnosis:")
    for d in dx:
        st.success(d)

    st.write("### 🔍 Differential diagnosis:")
    for d in ddx:
        st.info(d)

    # Suggested next steps
    st.subheader("📋 Suggested Next Steps (General)")
    st.write("""
    - Correlate clinically with vitals (SpO2, RR, fever)
    - CBC, CRP/ESR if infection suspected
    - Sputum AFB / GeneXpert if TB suspected
    - ECG + Echo if cardiomegaly/edema suspected
    - CT chest if unclear
    - Emergency referral if severe breathlessness / pneumothorax suspected
    """)

    # Plot bar chart
    st.subheader("📊 Probability Graph (Top 10 Findings)")
    top10 = sorted_findings[:10]
    names = [x[0] for x in top10]
    values = [x[1] for x in top10]

    fig, ax = plt.subplots()
    ax.barh(names[::-1], values[::-1])
    ax.set_xlabel("Probability")
    st.pyplot(fig)

else:
    st.info("Upload an X-ray to start analysis.")
