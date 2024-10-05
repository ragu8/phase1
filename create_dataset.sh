#!/bin/bash

# Function to delete datasets
delete_datasets() {
    rm -rf Binary_Dataset/ Multi_class_Dataset_1/ Multi_class_Dataset_2/
    echo "All datasets (Binary, Multi_class_Dataset_1, Multi_class_Dataset_2) deleted if they existed."
}

# Function to create binary dataset
create_binary_dataset() {
    rm -rf Augmented_DataSet/
    rm -rf Features/
    mkdir -p Binary_Dataset/Healthy
    mkdir -p Binary_Dataset/Reject
    
    cp -r Original_DataSet/Ripe/* Binary_Dataset/Healthy
    cp -r Original_DataSet/Unripe/* Binary_Dataset/Healthy
    cp -r Original_DataSet/Old/* Binary_Dataset/Reject
    cp -r Original_DataSet/Damaged/* Binary_Dataset/Reject

    echo "Binary Dataset creation Completed"
}

# Function to create multiclass dataset 1
create_multiclass_dataset_1() {
    rm -rf Augmented_DataSet/
    rm -rf Features/
    mkdir -p Multi_class_Dataset_1/Unripe
    mkdir -p Multi_class_Dataset_1/Ripe
    mkdir -p Multi_class_Dataset_1/Reject

    cp -r Original_DataSet/Ripe/* Multi_class_Dataset_1/Ripe
    cp -r Original_DataSet/Unripe/* Multi_class_Dataset_1/Unripe
    cp -r Original_DataSet/Old/* Multi_class_Dataset_1/Reject
    cp -r Original_DataSet/Damaged/* Multi_class_Dataset_1/Reject

    echo "Multiclass (Ripe/Unripe/Reject) Dataset creation Completed"
}

# Function to create multiclass dataset 2
create_multiclass_dataset_2() {
    rm -rf Augmented_DataSet/
    rm -rf Features/
    mkdir -p Multi_class_Dataset_2/Unripe
    mkdir -p Multi_class_Dataset_2/Ripe
    mkdir -p Multi_class_Dataset_2/Old
    mkdir -p Multi_class_Dataset_2/Damaged

    cp -r Original_DataSet/Ripe/* Multi_class_Dataset_2/Ripe
    cp -r Original_DataSet/Unripe/* Multi_class_Dataset_2/Unripe
    cp -r Original_DataSet/Old/* Multi_class_Dataset_2/Old
    cp -r Original_DataSet/Damaged/* Multi_class_Dataset_2/Damaged

    echo "Multiclass (Ripe/Unripe/Old/Damaged) Dataset creation Completed"
}

# Check the argument passed to the script
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 [binary|multiclass1|multiclass2|del]"
    exit 1
fi

case $1 in
    binary)
        create_binary_dataset
        ;;
    multiclass1)
        create_multiclass_dataset_1
        ;;
    multiclass2)
        create_multiclass_dataset_2
        ;;
    del)
        delete_datasets
        ;;
    *)
        echo "Invalid option: $1"
        echo "Usage: $0 [binary|multiclass1|multiclass2|del]"
        exit 1
        ;;
esac

# End of script
