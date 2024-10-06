import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

def plot_confusion_matrix(y_true, y_pred, feature_model, model, params, class_names):
    """Plot and save confusion matrix as PNG."""
    # Create the confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Create a folder for storing confusion matrices
    save_dir = 'Conf_Matrix'
    os.makedirs(save_dir, exist_ok=True)
    
    # Plotting the confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)

    # Title using feature_model + model + parameters
    title = f'{feature_model} + {model}'
    plt.title(title)
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')

    # Save plot as PNG
    save_path = os.path.join(save_dir, f'{feature_model}_{model}_{params}.png')
    plt.savefig(save_path)
    plt.close()

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
        'y_pred': y_pred,
        'time': time_taken,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1-score': f1_score
    }

def save_to_csv(feature_model, model, params, results, class_names):
    """Save the model results to a CSV file."""
    save_dir = 'Reports'
    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, 'model_results.csv')

    # Prepare data for saving
    data = {
        'Feature Model': feature_model,
        'Model': model,
        'Parameters': params,
        'Accuracy': results['accuracy'],
        'Precision': results['precision'],
        'Recall': results['recall'],
        'F1-Score': results['f1-score'],
        'Time Taken (s)': results['time'],
        'Class Names': ', '.join(class_names)  # Store class names as a string
    }

    # Convert data to DataFrame and append to CSV
    df = pd.DataFrame([data])
    if not os.path.exists(csv_path):
        df.to_csv(csv_path, index=False)
    else:
        df.to_csv(csv_path, mode='a', header=False, index=False)

def process_model(model_class, model_name, X_train, y_train, X_test, y_test, model_params, feature_model_name, class_names):
    """Process and evaluate the specified model."""
    model = model_class(**model_params)
    results = train_and_evaluate_model(model, X_train, y_train, X_test, y_test)
    
    # Get predictions
    y_pred = results['y_pred']
    
    # Format parameters as a string "key: value"
    params_str = ', '.join(f"{key}: {value}" for key, value in model_params.items())
    
    # Plot and save the confusion matrix
    plot_confusion_matrix(y_test, y_pred, feature_model_name, model_name, params_str, class_names)
    
    # Save evaluation results to CSV
    save_to_csv(feature_model_name, model_name, params_str, results, class_names)
    
    print(f"Model: {model_name} with parameters [{params_str}] Completed !")

def evaluate_models(model_classes=[SVC, DecisionTreeClassifier, RandomForestClassifier, KNeighborsClassifier], 
                    model_names=['SVC', 'Decision Tree', 'Random Forest', 'KNN'], 
                    Features_model_names=['ResNet50', 'InceptionV3', 'MobileNetV2', 'DenseNet121', 'EfficientNetB0'], 
                    save_dir='Features', 
                    test_size=0.2, 
                    class_names=None,  # Accept class_names as a parameter
                    model_params={}):
    if class_names is None:
        raise ValueError("Class names must be provided.")
    
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
                process_model(model_class, model_name, X_train_scaled, y_train, X_test_scaled, y_test, model_param, feature_model_name, class_names)


# # Example class names
# class_names = ["A", "B"]

# # Example model parameters for each classifier
# model_params = {
#     SVC: [{'C': 1, 'kernel': 'linear'}, {'C': 10, 'kernel': 'rbf'}],
#     DecisionTreeClassifier: [{'max_depth': 5}, {'max_depth': 10}],
#     RandomForestClassifier: [{'n_estimators': 100}, {'n_estimators': 200}],
#     KNeighborsClassifier: [{'n_neighbors': 3}, {'n_neighbors': 5}]
# }

# # Call the function to evaluate models and save confusion matrices and reports
# evaluate_models(class_names=class_names, model_params=model_params)
