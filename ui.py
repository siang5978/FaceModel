import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import requests


# Function to download the models from GitHub
@st.cache_resource(show_spinner=True)
def download_model(model_name):
    url = f'https://github.com/siang5978/FaceModel/raw/main/{model_name}.keras'  # Replace with your GitHub link
    response = requests.get(url)
    
    # Save the model to a local file
    model_path = f'{model_name}.keras'
    with open(model_path, 'wb') as file:
        file.write(response.content)
    
    return model_path

# Download the models
age_model_path = download_model('age_model')
gender_model_path = download_model('gender_model')
race_model_path = download_model('race_model')

# Load the models
age_model = load_model(age_model_path)
gender_model = load_model(gender_model_path)
race_model = load_model(race_model_path)

# Function to load and preprocess the image
def preprocess_image(image):
    target_size = (200, 200)  # Resize to the same size as used for training
    img = load_img(image, target_size=target_size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize the image
    return img_array

# Create Streamlit interface
st.title("Age, Gender, and Race Classification Model")

# Upload image through Streamlit
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Preprocess the image
    img_array = preprocess_image(uploaded_image)

    # Display the uploaded image
    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

    # Make predictions for age, gender, and race
    age_prediction = age_model.predict(img_array)
    gender_prediction = gender_model.predict(img_array)
    race_prediction = race_model.predict(img_array)

    # Interpret results
    age_groups = ['0-8', '9-18', '19-50', '50+']
    gender_classes = ['Male', 'Female']
    race_classes = ['White', 'Black', 'Asian', 'Indian']

    # Get the predicted age group, gender, and race
    predicted_age = age_groups[np.argmax(age_prediction)]
    predicted_gender = gender_classes[round(gender_prediction[0][0])]  # Binary prediction
    predicted_race = race_classes[np.argmax(race_prediction)]

    # Display the results
    st.write(f"Predicted Age Group: **{predicted_age}**")
    st.write(f"Predicted Gender: **{predicted_gender}**")
    st.write(f"Predicted Race: **{predicted_race}**")
