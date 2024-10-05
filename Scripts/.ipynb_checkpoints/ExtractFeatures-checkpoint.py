import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50, InceptionV3, MobileNetV2, DenseNet121, EfficientNetB0,VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model

def extract_features(model_name, data_dir, img_height=150, img_width=150, batch_size=32, save_dir='Features'):
    # Create the directory for saving features if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Dictionary to map model names to Keras model functions
    model_dict = {
        'ResNet50': ResNet50,
        'InceptionV3': InceptionV3,
        'MobileNetV2': MobileNetV2,
        'DenseNet121': DenseNet121,
        'EfficientNetB0': EfficientNetB0,
        'VGG16': VGG16
    }

    # Check if the model_name is valid
    if model_name not in model_dict:
        raise ValueError(f"Model name '{model_name}' is not valid. Choose from {list(model_dict.keys())}.")

    # Load the selected model without the top layers
    base_model = model_dict[model_name](weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))

    # Define the model for feature extraction
    feature_extractor = Model(inputs=base_model.input, outputs=base_model.output)

    # Create an ImageDataGenerator for loading and preprocessing the images
    datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    # Load images for feature extraction with labels
    feature_generator = datagen.flow_from_directory(
        data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical',  # Set class_mode to 'categorical'
        shuffle=False  # Shuffle is not necessary for feature extraction
    )

    # Calculate the number of steps based on the generator
    steps = feature_generator.samples // batch_size + 1

    # Extract features using the selected model
    extracted_features = feature_extractor.predict(feature_generator, steps=steps)

    # Get the class labels from the generator
    labels = feature_generator.classes

    # Save the extracted features and labels to files in the specified directory
    np.save(os.path.join(save_dir, f'extracted_features_{model_name.lower()}.npy'), extracted_features)
    np.save(os.path.join(save_dir, f'extracted_labels_{model_name.lower()}.npy'), labels)

    print(f"Features and labels extracted using {model_name} and saved successfully in '{save_dir}' directory.")
