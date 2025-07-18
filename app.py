import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from PIL import Image, ImageOps

# Load the saved Random Forest model
model = joblib.load("random_forest_mnist.pkl")

# Title
st.title("🧠 Handwritten Digit Classifier (Random Forest)")
st.markdown("Upload an image of a digit (28x28 grayscale) or draw your own!")

# File uploader
uploaded_file = st.file_uploader("Choose a digit image (PNG or JPG)", type=["png", "jpg", "jpeg"])

def preprocess_image(image):
    # Convert to grayscale
    image = image.convert("L")
    # Resize to 28x28 if not already
    image = image.resize((28, 28))
    # Invert colors: background black, digit white
    image = ImageOps.invert(image)
    # Convert to numpy array and flatten
    img_array = np.array(image).reshape(1, -1)
    return img_array

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Digit", use_column_width=False, width=150)
    
    # Preprocess and predict
    processed_img = preprocess_image(image)
    prediction = model.predict(processed_img)[0]
    st.success(f"🎯 Predicted Digit: **{prediction}**")
    
    # Show pixel-level view
    if st.checkbox("Show Pixel Grid"):
        plt.imshow(processed_img.reshape(28, 28), cmap='gray')
        plt.title("Preprocessed Image")
        st.pyplot(plt.gcf())
