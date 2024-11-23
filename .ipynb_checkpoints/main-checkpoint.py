import os
import shutil
import glob
import logging
import subprocess
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from Scripts.ImageAug import process_dataset  # Assuming process_dataset function is defined in ImageAug
from Scripts.ExtractFeatures import extract_features  # Assuming extract_features function is defined in ExtractFeatures
from Scripts.Models import evaluate_models  # Assuming evaluate_models function is defined in Models


# In[4]:


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# In[5]:


def list_directory_contents(directory):
    """List the contents of a directory."""
    logging.info(f"Contents of {directory}:")
    contents = os.listdir(directory)
    for item in contents:
        logging.info(item)


# In[6]:


def create_and_process_dataset(dataset_type, num_augmentations=1):
    """Create and process the dataset."""
    os.system(f"./Scripts/create_dataset.sh {dataset_type}")
    logging.info("Dataset creation script executed for %s", dataset_type)
    
    dataset_dir = f"{dataset_type.capitalize()}_Dataset/"
    
    # Check if the dataset directory was created
    if not os.path.exists(dataset_dir):
        logging.error("Dataset directory not found: %s", dataset_dir)
        raise FileNotFoundError(f"Dataset directory '{dataset_dir}' does not exist.")
    
    list_directory_contents(dataset_dir)
    
    process_dataset(dataset_dir, num_augmentations=num_augmentations)
    logging.info("Processing completed for %s", dataset_dir)
    
    return 'Augmented_DataSet/'


# In[7]:


def extract_features_for_models(data_dir):
    """Extract features using various models."""
    models = ['ResNet50', 'InceptionV3', 'MobileNetV2', 'DenseNet121', 'EfficientNetB0']
    for model in models:
        extract_features(model, data_dir)
        logging.info("Extracted features using %s", model)


# In[8]:


def evaluate_models_and_save_results(model_params, output_filename, class_names, dataset_type):
    """Evaluate models, save results, and save models to directory."""
    results = evaluate_models(model_params=model_params, class_names=class_names)
    
    # Ensure Reports directory exists before renaming
    if not os.path.exists("Reports"):
        os.makedirs("Reports")
    
    # Save the results to the specified output filename
    os.rename("Reports/model_results.csv", output_filename)
    logging.info("Model results saved to %s", output_filename)

    # Move saved models to specific directory for this dataset type
    move_models(dataset_type)


# In[9]:


def move_cm(dir_name):
    """Move confusion matrices to a directory named after the dataset type."""
    os.makedirs(f"Conf_Matrix/{dir_name}", exist_ok=True)
    for file_path in glob.glob("Conf_Matrix/*.png"):
        shutil.move(file_path, f"Conf_Matrix/{dir_name}/")
        print(f'Moved: {file_path} to Conf_Matrix/{dir_name}/')


# In[10]:


def move_models(dir_name):
    """Move saved models to a directory named after the dataset type."""
    os.makedirs(f"Models/{dir_name}", exist_ok=True)
    for file_path in glob.glob("Models/*.pkl"):  # Assuming models are saved as .pkl files
        shutil.move(file_path, f"Models/{dir_name}/")
        print(f'Moved: {file_path} to Models/{dir_name}/')


# In[11]:


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

    # Dataset configurations: dataset type, result filename, and class names for each type
    datasets = [
        {'type': 'binary', 'filename': 'Reports/model_results_binary.csv', 'class_names': ['Healthy', 'Reject']},
        {'type': 'multiclass1', 'filename': 'Reports/model_results_three_class.csv', 'class_names': ['Ripe', 'Unripe', 'Reject']},
        {'type': 'multiclass2', 'filename': 'Reports/model_results_four_class.csv', 'class_names': ['Ripe', 'Unripe', 'Old', 'Damaged']}
    ]

    # Create, process datasets, extract features, evaluate models, and organize results
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
    # Command to remove the specified directories
    command = "rm -rf Augmented_DataSet Binary_Dataset Features Multiclass1_Dataset Multiclass2_Dataset"

    # Execute the command in a shell
    try:
        subprocess.run(command, shell=True, check=True)
        print("Directories removed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred: {e}")


# In[ ]:


if __name__ == "__main__":
    main()
