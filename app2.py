import streamlit as st
import cv2
import numpy as np
import joblib
from PIL import Image
from skimage.feature import hog

# ---- LOAD MODEL ----
model = joblib.load("best_logistic_model2.pkl")

# Emotion labels (JAFFE)
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# HOG feature extraction
def extract_hog(img):
    img = cv2.resize(img, (96, 96))
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features = hog(
        img,
        orientations=9,
        pixels_per_cell=(8,8),
        cells_per_block=(2,2),
        block_norm="L2-Hys"
    )
    return features

# ---- STREAMLIT UI ----
st.set_page_config(page_title="Facial Emotion Recognition", layout="centered")
st.title("😃 Facial Emotion Recognition - JAFFE (Logistic Regression)")

uploaded_file = st.file_uploader("Upload an image (TIFF, JPG, PNG)", type=["tiff", "tif", "jpg", "png"])

if uploaded_file is not None:
    # Read image with PIL and convert to NumPy
    img = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(img)

    # Show uploaded image
    st.image(img_np, caption="Uploaded Image", use_column_width=True)

    # Extract features & predict
    features = extract_hog(img_np).reshape(1, -1)
    prediction = model.predict(features)[0]
    prob = model.predict_proba(features)[0]

    st.subheader("Predicted Emotion:")
    st.write(emotion_labels[prediction])

    st.subheader("Probabilities:")
    probs_dict = {emotion_labels[i]: f"{prob[i]*100:.2f}%" for i in range(len(emotion_labels))}
    st.json(probs_dict)
