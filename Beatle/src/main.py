import os
import torch
from ultralytics import YOLO

# Function to clear the labels.cache file
def clear_labels_cache(cache_path):
    if os.path.exists(cache_path):
        try:
            os.remove(cache_path)
            print(f"Successfully removed: {cache_path}")
        except Exception as e:
            print(f"Error removing {cache_path}: {e}")
    else:
        print(f"File does not exist: {cache_path}")

# Define the path to the labels.cache file
labels_cache_path = "/home/christiaan/Documents/GitHub/STARRGazer/Beatle/seg/valid/labels.cache"

# Clear the labels.cache file
clear_labels_cache(labels_cache_path)

# Define the path to the models directory
model_path = "../models/"

# Define the path to the dataset YAML file
data_yaml_path = "../seg/data.yaml"

# Define the list of YOLOv11 model types
model_types = ['yolo11s', 'yolo11m', 'yolo11l', 'yolo11x']

# Iterate through each model type, train it, and save results with unique names
for model_type in model_types:
    print(f"Training model: {model_type}")
    
    # Initialize the YOLO model with the respective weights
    selected_model = YOLO(f"{model_path}{model_type}.pt")
    
    # Train the selected model
    selected_model.train(
        data=data_yaml_path,  # Path to the data.yaml file
        epochs=2,            # Number of epochs (adjust as needed)
        imgsz=640,            # Image size (adjust as needed)
        batch=12,             # Batch size (adjust as needed)
        name=f"{model_type}_training",  # Name of the training run
        resume=False          # Ensure training starts fresh
    )
    
    # Evaluate the model on the test set
    print(f"Evaluating model: {model_type}")
    results = selected_model.val(data=data_yaml_path)
    print(f"Test results for {model_type}:", results)

    # Save the model with a unique name
    saved_model_path = f"{model_path}{model_type}_trained.pt"
    selected_model.save(saved_model_path)
    print(f"Model saved as: {saved_model_path}")

    # Free up resources
    del selected_model
    torch.cuda.empty_cache()

print("All models trained and saved successfully.")
