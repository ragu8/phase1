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
    mkdir -p Multiclass1_Dataset/Unripe
    mkdir -p Multiclass1_Dataset/Ripe
    mkdir -p Multiclass1_Dataset/Reject

    cp -r Original_DataSet/Ripe/* Multiclass1_Dataset/Ripe
    cp -r Original_DataSet/Unripe/* Multiclass1_Dataset/Unripe
    cp -r Original_DataSet/Old/* Multiclass1_Dataset/Reject
    cp -r Original_DataSet/Damaged/* Multiclass1_Dataset/Reject

    echo "Multiclass (Ripe/Unripe/Reject) Dataset creation Completed"
}

# Function to create multiclass dataset 2
create_multiclass_dataset_2() {
    rm -rf Augmented_DataSet/
    rm -rf Features/
    mkdir -p Multiclass2_Dataset/Unripe
    mkdir -p Multiclass2_Dataset/Ripe
    mkdir -p Multiclass2_Dataset/Old
    mkdir -p Multiclass2_Dataset/Damaged

    cp -r Original_DataSet/Ripe/* Multiclass2_Dataset/Ripe
    cp -r Original_DataSet/Unripe/* Multiclass2_Dataset/Unripe
    cp -r Original_DataSet/Old/* Multiclass2_Dataset/Old
    cp -r Original_DataSet/Damaged/* Multiclass2_Dataset/Damaged

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
