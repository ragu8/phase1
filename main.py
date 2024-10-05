import os
import logging
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from Scripts.ImageAug import process_dataset
from Scripts.ExtractFeatures import extract_features
from Scripts.Models import evaluate_models

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def list_directory_contents(directory):
    """List the contents of a directory."""
    logging.info(f"Contents of {directory}:")
    contents = os.listdir(directory)
    for item in contents:
        logging.info(item)

def create_and_process_dataset(dataset_type, num_augmentations=2):
    """Create and process the dataset."""
    os.system(f"./create_dataset.sh {dataset_type}")
    dataset_dir = f"{dataset_type.capitalize()}_Dataset/"
    list_directory_contents(dataset_dir)
    
    process_dataset(dataset_dir, num_augmentations=num_augmentations)
    logging.info("Processing completed for %s", dataset_dir)
    
    return 'Augmented_DataSet/'

def extract_features_for_models(data_dir):
    """Extract features using various models."""
    models = ['ResNet50', 'InceptionV3', 'MobileNetV2', 'DenseNet121', 'EfficientNetB0', 'VGG16']
    for model in models:
        extract_features(model, data_dir)
        logging.info("Extracted features using %s", model)

def evaluate_models_and_save_results(model_params, output_filename):
    """Evaluate models and save results."""
    results = evaluate_models(model_params=model_params)
    os.rename("Reports/model_results.csv", output_filename)
    logging.info("Model results saved to %s", output_filename)

def main():
    # List contents of Original_DataSet
    list_directory_contents("Original_DataSet/")

    # Define model parameters
    model_params = {
        SVC: [{'kernel': 'rbf'}, {'kernel': 'linear'}],
        DecisionTreeClassifier: [{'max_depth': None}, {'max_depth': 5}],
        RandomForestClassifier: [{'n_estimators': 100, 'max_depth': None, 'random_state': 42}],
        KNeighborsClassifier: [{'n_neighbors': 5}]
    }

    # Create and process datasets
    for dataset_type, result_filename in zip(['binary', 'multiclass1', 'multiclass2'], 
                                             ['model_results_binary.csv', 'model_results_three_class.csv', 'model_results_four_class.csv']):
        data_dir = create_and_process_dataset(dataset_type)
        extract_features_for_models(data_dir)
        evaluate_models_and_save_results(model_params, result_filename)

    logging.info("#################### Completed ##########################")

if __name__ == "__main__":
    main()
