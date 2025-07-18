import streamlit as st
import numpy as np
import joblib
import os
import gdown
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

# Google Drive file ID for your Random Forest model
file_id = '1kb9wKJL1knwqK7mvLPFLXPIAmbZhmse4'
url = f"https://drive.google.com/uc?id={file_id}"
model_path = "random_forest_mnist.pkl"

# Download the model if not present locally
if not os.path.exists(model_path):
    with st.spinner("⬇️ Downloading model from Google Drive..."):
        gdown.download(url, model_path, quiet=False)
        st.success("✅ Model downloaded successfully!")

# Load the Random Forest model
model = joblib.load(model_path)

# Title and description
st.title("🧠 Handwritten Digit Classifier")
st.markdown("Upload a **28x28 grayscale** digit image (like from MNIST) and the model will predict the digit.")

# Upload digit image
uploaded_file = st.file_uploader("Upload a digit image (PNG, JPG)", type=["png", "jpg", "jpeg"])

# Function to process image
def preprocess_image(image):
    image = image.convert("L")  # Convert to grayscale
    image = image.resize((28, 28))  # Resize to 28x28
    image = ImageOps.invert(image)  # Invert for white digits on black bg
    img_array = np.array(image).reshape(1, -1)
    return img_array

# Predict if image is uploaded
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="📷 Uploaded Image", width=150)

    processed_img = preprocess_image(image)
    prediction = model.predict(processed_img)[0]
    st.success(f"🎯 Predicted Digit: **{prediction}**")

    if st.checkbox("Show Preprocessed Image"):
        plt.imshow(processed_img.reshape(28, 28), cmap='gray')
        plt.axis('off')
        st.pyplot(plt.gcf())
