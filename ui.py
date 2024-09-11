{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "d0f58436-9cda-477f-adce-e4b81282d386",
   "metadata": {},
   "outputs": [],
   "source": [
    "import streamlit as st\n",
    "from tensorflow.keras.models import load_model\n",
    "from tensorflow.keras.preprocessing.image import load_img, img_to_array\n",
    "import numpy as np\n",
    "\n",
    "# Load the pre-trained model (make sure the path is correct)\n",
    "model = load_model('path_to_your_model/race_model.h5')\n",
    "\n",
    "# Function to load and preprocess the image\n",
    "def preprocess_image(image):\n",
    "    target_size = (200, 200)  # Resize to the same size as used for training\n",
    "    img = load_img(image, target_size=target_size)\n",
    "    img_array = img_to_array(img)\n",
    "    img_array = np.expand_dims(img_array, axis=0)\n",
    "    img_array /= 255.0  # Normalize the image\n",
    "    return img_array\n",
    "\n",
    "# Create Streamlit interface\n",
    "st.title(\"Race Classification Model\")\n",
    "\n",
    "# Upload image through Streamlit\n",
    "uploaded_image = st.file_uploader(\"Upload an image\", type=[\"jpg\", \"jpeg\", \"png\"])\n",
    "\n",
    "if uploaded_image is not None:\n",
    "    # Preprocess the image\n",
    "    img_array = preprocess_image(uploaded_image)\n",
    "\n",
    "    # Display the uploaded image\n",
    "    st.image(uploaded_image, caption=\"Uploaded Image\", use_column_width=True)\n",
    "\n",
    "    # Make prediction\n",
    "    predictions = model.predict(img_array)\n",
    "    race_classes = ['White', 'Black', 'Asian', 'Indian']\n",
    "\n",
    "    # Get the predicted race\n",
    "    predicted_race = race_classes[np.argmax(predictions)]\n",
    "\n",
    "    # Display the result\n",
    "    st.write(f\"Predicted Race: **{predicted_race}**\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "0e4a0e4b-7ea2-4d31-9ad2-5173dfc7d293",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
