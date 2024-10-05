import os
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

def train_and_evaluate_model(model, X_train, y_train, X_test, y_test):
    """Train the model and evaluate its performance."""
    start_time = time.time()
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Predict on the test set
    y_pred = model.predict(X_test)

    # Measure time taken
    time_taken = time.time() - start_time

    # Evaluate the model
    accuracy = np.mean(y_pred == y_test)
    precision = np.sum((y_pred == 1) & (y_test == 1)) / np.sum(y_pred == 1) if np.sum(y_pred == 1) != 0 else 0
    recall = np.sum((y_pred == 1) & (y_test == 1)) / np.sum(y_test == 1) if np.sum(y_test == 1) != 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
    
    return {
        'time': time_taken,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1-score': f1_score
    }

def process_model(model_class, model_name, X_train, y_train, X_test, y_test, model_params, feature_model_name):
    """Process and evaluate the specified model."""
    model = model_class(**model_params)
    results = train_and_evaluate_model(model, X_train, y_train, X_test, y_test)
    results['feature_model'] = feature_model_name  
    results['model'] = model_name
    
    # Format parameters as a string "key: value"
    params_str = ', '.join(f"{key}: {value}" for key, value in model_params.items())
    results['parameters'] = params_str  # Store formatted parameters string
    
    print(f"Model: {model_name} with parameters [{params_str}] Completed !")
    return results

def evaluate_models(model_classes=[SVC, DecisionTreeClassifier, RandomForestClassifier, KNeighborsClassifier], model_names=['SVC', 'Decision Tree', 'Random Forest', 'KNN'], Features_model_names=['ResNet50', 'InceptionV3', 'MobileNetV2', 'DenseNet121', 'EfficientNetB0'], save_dir='Features', test_size=0.2, model_params={}):
    results = []
    
    for feature_model_name in Features_model_names:
        # Load the extracted features and labels
        features_file = f'{save_dir}/extracted_features_{feature_model_name.lower()}.npy'
        labels_file = f'{save_dir}/extracted_labels_{feature_model_name.lower()}.npy'

        # Check if files exist
        if not os.path.exists(features_file):
            print(f"Features file not found: {features_file}")
            continue
        if not os.path.exists(labels_file):
            print(f"Labels file not found: {labels_file}")
            continue

        extracted_features = np.load(features_file)
        extracted_labels = np.load(labels_file)

        # Debugging: Check the shapes of features and labels
        print(f"\n Loaded {feature_model_name} Features ! \n")

        # Flatten the extracted features
        extracted_features_flattened = extracted_features.reshape(extracted_features.shape[0], -1)

        # Split the dataset into training and testing sets (80% train, 20% test)
        X_train, X_test, y_train, y_test = train_test_split(
            extracted_features_flattened, 
            extracted_labels, 
            test_size=test_size, 
            random_state=42,
            stratify=extracted_labels  # Ensure balanced class distribution
        )

        # Standardize the features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Process each model with each set of parameters
        for model_class, model_name in zip(model_classes, model_names):
            for model_param in model_params.get(model_class, []):
                results.append(process_model(model_class, model_name, X_train_scaled, y_train, X_test_scaled, y_test, model_param, feature_model_name))

    # Create the Reports directory if it doesn't exist
    reports_dir = 'Reports'
    os.makedirs(reports_dir, exist_ok=True)

    # Save results to a CSV file
    results_df = pd.DataFrame(results)
    results_csv_path = os.path.join(reports_dir, 'model_results.csv')
    results_df.to_csv(results_csv_path, index=False)

    return results