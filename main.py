import os
import shutil
import glob
import logging
import subprocess
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from Scripts.ImageAug import process_dataset  
from Scripts.ExtractFeatures import extract_features  
from Scripts.Models import evaluate_models  



logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')





def list_directory_contents(directory):
    """List the contents of a directory."""
    logging.info(f"Contents of {directory}:")
    contents = os.listdir(directory)
    for item in contents:
        logging.info(item)





def create_and_process_dataset(dataset_type, num_augmentations=1):
    """Create and process the dataset."""
    os.system(f"./Scripts/create_dataset.sh {dataset_type}")
    logging.info("Dataset creation script executed for %s", dataset_type)
    
    dataset_dir = f"{dataset_type.capitalize()}_Dataset/"
    
    if not os.path.exists(dataset_dir):
        logging.error("Dataset directory not found: %s", dataset_dir)
        raise FileNotFoundError(f"Dataset directory '{dataset_dir}' does not exist.")
    
    list_directory_contents(dataset_dir)
    
    process_dataset(dataset_dir, num_augmentations=num_augmentations)
    logging.info("Processing completed for %s", dataset_dir)
    
    return 'Augmented_DataSet/'





def extract_features_for_models(data_dir):
    """Extract features using various models."""
    models = ['ResNet50', 'InceptionV3', 'MobileNetV2', 'DenseNet121', 'EfficientNetB0']
    for model in models:
        extract_features(model, data_dir)
        logging.info("Extracted features using %s", model)





def evaluate_models_and_save_results(model_params, output_filename, class_names, dataset_type):
    """Evaluate models, save results, and save models to directory."""
    results = evaluate_models(model_params=model_params, class_names=class_names)
    
    if not os.path.exists("Reports"):
        os.makedirs("Reports")
    
    os.rename("Reports/model_results.csv", output_filename)
    logging.info("Model results saved to %s", output_filename)

    move_models(dataset_type)




def move_cm(dir_name):
    """Move confusion matrices to a directory named after the dataset type."""
    os.makedirs(f"Conf_Matrix/{dir_name}", exist_ok=True)
    for file_path in glob.glob("Conf_Matrix/*.png"):
        shutil.move(file_path, f"Conf_Matrix/{dir_name}/")
        print(f'Moved: {file_path} to Conf_Matrix/{dir_name}/')




def move_models(dir_name):
    """Move saved models to a directory named after the dataset type."""
    os.makedirs(f"Models/{dir_name}", exist_ok=True)
    for file_path in glob.glob("Models/*.pkl"):  
        shutil.move(file_path, f"Models/{dir_name}/")
        print(f'Moved: {file_path} to Models/{dir_name}/')




def main():
    list_directory_contents("Original_DataSet/")


    model_params = {
    SVC: [
        {'kernel': 'rbf', 'C': 0.1, 'gamma': 'scale'},
        {'kernel': 'rbf', 'C': 0.1, 'gamma': 'auto'},
        {'kernel': 'rbf', 'C': 1, 'gamma': 'scale'},
        {'kernel': 'rbf', 'C': 1, 'gamma': 'auto'},
        {'kernel': 'rbf', 'C': 10, 'gamma': 'scale'},
        {'kernel': 'rbf', 'C': 10, 'gamma': 'auto'},
        {'kernel': 'linear', 'C': 0.1},
        {'kernel': 'linear', 'C': 1},
        {'kernel': 'linear', 'C': 10}
    ],
    DecisionTreeClassifier: [
        {'max_depth': None, 'min_samples_split': 2},
        {'max_depth': None, 'min_samples_split': 5},
        {'max_depth': None, 'min_samples_split': 10},
        {'max_depth': 5, 'min_samples_split': 2},
        {'max_depth': 5, 'min_samples_split': 5},
        {'max_depth': 5, 'min_samples_split': 10}
    ],
    RandomForestClassifier: [
        {'n_estimators': 100, 'max_depth': None, 'random_state': 42, 'max_features': 'sqrt'},
        {'n_estimators': 100, 'max_depth': None, 'random_state': 42, 'max_features': 'log2'},
        {'n_estimators': 200, 'max_depth': 10, 'min_samples_split': 2, 'max_features': 'sqrt'},
        {'n_estimators': 200, 'max_depth': 10, 'min_samples_split': 5, 'max_features': 'sqrt'},
        {'n_estimators': 200, 'max_depth': None, 'min_samples_split': 2, 'max_features': 'sqrt'},
        {'n_estimators': 200, 'max_depth': None, 'min_samples_split': 5, 'max_features': 'sqrt'}
    ],
    KNeighborsClassifier: [
        {'n_neighbors': 3, 'weights': 'uniform'},
        {'n_neighbors': 3, 'weights': 'distance'},
        {'n_neighbors': 5, 'weights': 'uniform'},
        {'n_neighbors': 5, 'weights': 'distance'},
        {'n_neighbors': 7, 'weights': 'uniform'},
        {'n_neighbors': 7, 'weights': 'distance'}
    ]
    }


    datasets = [
        {'type': 'binary', 'filename': 'Reports/model_results_binary.csv', 'class_names': ['Healthy', 'Reject']},
        {'type': 'multiclass1', 'filename': 'Reports/model_results_three_class.csv', 'class_names': ['Ripe', 'Unripe', 'Reject']},
        {'type': 'multiclass2', 'filename': 'Reports/model_results_four_class.csv', 'class_names': ['Ripe', 'Unripe', 'Old', 'Damaged']}
    ]

    for dataset in datasets:
        data_dir = create_and_process_dataset(dataset['type'])
        extract_features_for_models(data_dir)
        evaluate_models_and_save_results(
            model_params=model_params,
            output_filename=dataset['filename'],
            class_names=dataset['class_names'],
            dataset_type=dataset['type']
        )
        move_cm(dataset['type'])

    logging.info("#################### Completed ##########################")
    command = "rm -rf Augmented_DataSet Binary_Dataset Features Multiclass1_Dataset Multiclass2_Dataset"

    try:
        subprocess.run(command, shell=True, check=True)
        print("Directories removed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred: {e}")




if __name__ == "__main__":
    main()

