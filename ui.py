import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import requests
import os

# Function to download the model from GitHub
@st.cache_resource(show_spinner=True)
def download_model():
    url = 'https://github.com/siang5978/FaceModel/raw/main/race_model.keras'  # Replace with your GitHub link
    response = requests.get(url)
    
    # Save the model to a local file
    model_path = 'race_model.keras'
    with open(model_path, 'wb') as file:
        file.write(response.content)
    
    return model_path

# Download the model
model_path = download_model()
model = load_model(model_path)

# Function to load and preprocess the image
def preprocess_image(image):
    target_size = (200, 200)  # Resize to the same size as used for training
    img = load_img(image, target_size=target_size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize the image
    return img_array

# Create Streamlit interface
st.title("Race Classification Model")

# Upload image through Streamlit
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Preprocess the image
    img_array = preprocess_image(uploaded_image)

    # Display the uploaded image
    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

    # Make prediction
    predictions = model.predict(img_array)
    race_classes = ['White', 'Black', 'Asian', 'Indian']

    # Get the predicted race
    predicted_race = race_classes[np.argmax(predictions)]

    # Display the result
    st.write(f"Predicted Race: **{predicted_race}**")
