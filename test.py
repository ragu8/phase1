import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
from sklearn.svm import SVC
import pickle
import matplotlib.pyplot as plt

# Load the trained SVC model
with open('Models/multiclass2/ResNet50_SVC_kernel: linear.pkl', 'rb') as f:  # Using 'with' to open the file
    svc = pickle.load(f)

categories = ['Damaged', 'Old', 'Ripe', 'Unripe']

base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

def predict_single_image(img_path, base_model, svc, categories):
    # Load and preprocess the image
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Extract features using ResNet50
    features = base_model.predict(img_array)
    features = features.flatten() 

 
    if features.size != 51200:
        features = features[:51200]  # Truncate or reshape to the expected size

    prediction = svc.predict(features.reshape(1, -1))  # Reshape for SVC input
    predicted_class = prediction[0]
    
    # Map the prediction back to the corresponding category
    predicted_label = categories[predicted_class]  # Map to category name

    return predicted_label

single_image_path = 'Original_DataSet/Damaged/D016.png'  
predicted_label = predict_single_image(single_image_path, base_model, svc, categories)
print(f"The predicted label for the image is: {predicted_label}")
