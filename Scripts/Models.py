import os
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import matplotlib.pyplot as plt

def svm(model_names=['ResNet50', 'InceptionV3', 'MobileNetV2', 'DenseNet121', 'EfficientNetB0'], 
                                save_dir='Features', kernel='rbf'):
    results = []
    
    for model_name in model_names:
        # Load the extracted features and labels
        features_file = f'{save_dir}/extracted_features_{model_name.lower()}.npy'
        labels_file = f'{save_dir}/extracted_labels_{model_name.lower()}.npy'

        # Check if files exist
        if not os.path.exists(features_file):
            print(f"Features file not found: {features_file}")
            continue
        if not os.path.exists(labels_file):
            print(f"Labels file not found: {labels_file}")
            continue

        extracted_features = np.load(features_file)
        extracted_labels = np.load(labels_file)

        # Flatten the extracted features (if they are 3D) for SVM
        extracted_features_flattened = extracted_features.reshape(extracted_features.shape[0], -1)

        # Split the dataset into training and testing sets (80% train, 20% test)
        X_train, X_test, y_train, y_test = train_test_split(
            extracted_features_flattened, 
            extracted_labels, 
            test_size=0.2, 
            random_state=42,
            stratify=extracted_labels  # Ensure balanced class distribution
        )

        # Standardize the features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Measure time taken for training
        start_time = time.time()
        
        # Create and train the SVM model
        svm_model = SVC(kernel=kernel)  
        svm_model.fit(X_train_scaled, y_train)

        # Predict on the test set
        y_pred = svm_model.predict(X_test_scaled)

        # Measure time taken
        time_taken = time.time() - start_time

        # Evaluate the model
        accuracy = np.mean(y_pred == y_test)
        precision = np.sum((y_pred == 1) & (y_test == 1)) / np.sum(y_pred == 1) if np.sum(y_pred == 1) != 0 else 0
        recall = np.sum((y_pred == 1) & (y_test == 1)) / np.sum(y_test == 1) if np.sum(y_test == 1) != 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
        print(f"Model: {model_name} Completed ! ")
        # Store results
        results.append({
            'model': model_name,
            'time': time_taken,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1-score': f1_score
        })

    # Create the Reports directory if it doesn't exist
    reports_dir = 'Reports'
    os.makedirs(reports_dir, exist_ok=True)

    # Save results to a CSV file
    results_df = pd.DataFrame(results)
    results_csv_path = os.path.join(reports_dir, f'svm_model_{kernel}_results.csv')
    results_df.to_csv(results_csv_path, index=False)

    return results