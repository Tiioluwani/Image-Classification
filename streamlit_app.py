import streamlit as st
from predict import predict_image
from PIL import Image

st.title("Intel Image Classifier")
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)
    img_path = "temp.jpg"
    img.save(img_path)

    label = predict_image(img_path)
    st.success(f"Predicted Class: **{label}**")
