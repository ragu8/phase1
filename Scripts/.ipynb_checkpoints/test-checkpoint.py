import re
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ( ResNet50, InceptionV3, MobileNetV2, DenseNet121, EfficientNetB0, VGG16)
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
import pickle

def load_pretrained(path, weights='imagenet', include_top=False, input_shape=(224, 224, 3)):

    models = {
        "ResNet50": ResNet50,
        "InceptionV3": InceptionV3,
        "MobileNetV2": MobileNetV2,
        "DenseNet121": DenseNet121,
        "EfficientNetB0": EfficientNetB0,
        }

    features_size = {
    "ResNet50": 51200,
    "InceptionV3": 18432,
    "MobileNetV2": 32000,
    "DenseNet121": 16384,
    "EfficientNetB0": 32000
}
    model_name = None
    for key in models.keys():
        if key in path:
            model_name = key
            print(f"model_loaded : {model_name}")
            break

    if not model_name:
        raise ValueError("No valid model name found in the given path.")

    return models[model_name](weights=weights, include_top=include_top, input_shape=input_shape),features_size[model_name]



def load_categories(path):
    category_map = {
        'binary': ['Healthy', 'Reject'],
        'multiclass1': ['Reject', 'Ripe', 'Unripe'],
        'multiclass2': ['Damaged', 'Old', 'Ripe', 'Unripe']
    }
    
    for key in category_map:
        if key in path.lower():
            print(f"categories : {category_map[key]}")
            return category_map[key]
    
    raise ValueError("No matching category found for the given path.")



def load_classifier(file_path):
   
    file_name = file_path.split("/")[-1]
    
    pattern = r'(?P<model_name>[A-Za-z0-9_]+)_(?P<classifier>[A-Za-z\s]+)(?:_(?P<hyperparams>.+))?\.pkl'
    
    match = re.match(pattern, file_name)
    
    if not match:
        raise ValueError("Invalid file format or unable to extract details.")
    
    model_details = {
        "model_name": match.group("model_name"),
        "classifier": match.group("classifier"),
        "hyperparameters": match.group("hyperparams") or "None"
    }
    
    try:
        with open(file_path, 'rb') as f:
            classifier = pickle.load(f)
        print(f"Classifier : {model_details['classifier']}")
        print(f"hyperparameters : {model_details['hyperparameters']}")
    except FileNotFoundError:
        raise FileNotFoundError(f"Error: File not found at {file_path}")
    except pickle.UnpicklingError:
        raise ValueError("Error: Unable to unpickle the object. Ensure the file is a valid pickle file.")
    except Exception as e:
        raise RuntimeError(f"An unexpected error occurred: {e}")
    
    model_details["classifier_object"] = classifier
    return classifier


def predict_image(img_path, pretrained_model, classifier, categories,features_size):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    features = pretrained_model.predict(img_array)
    features = features.flatten() 
    if features.size != features_size:
        features = features[:features_size]  

    prediction = classifier.predict(features.reshape(1, -1))  
    predicted_class = prediction[0]
    predicted_label = categories[predicted_class]  

    return predicted_label
