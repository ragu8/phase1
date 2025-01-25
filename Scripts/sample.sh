#!/bin/bash

# Define the source and destination directories
SOURCE_DIR="Original_DataSe"
DEST_DIR="Original_DataSet"

# Create the destination directory if it doesn't exist
mkdir -p "$DEST_DIR"

# Loop through each category in the source directory
for category in Damaged Old Ripe Unripe; do
  # Create the same structure in the destination directory
  mkdir -p "$DEST_DIR/$category"
  
  # Copy the first 10 images from each category
  find "$SOURCE_DIR/$category" -type f | head -n 10 | while read image; do
    cp "$image" "$DEST_DIR/$category/"
  done
done

echo "Original_DataSet created with 10 images from each category."
