import os
import torch
import csv
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

# Define the list of YOLO model types (including YOLOv5, YOLOv8, and YOLOv11 with nano models)
model_types = [
    'yolov5n', 'yolov5s', 'yolov5m', 'yolov5l', 'yolov5x',
    'yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x',
    'yolo11n', 'yolo11s', 'yolo11m', 'yolo11l', 'yolo11x'
]

# Iterate through each model type, perform hyperparameter tuning, train, evaluate, and save results with unique names
results_summary = []

for model_type in model_types:
    print(f"Tuning hyperparameters for model: {model_type}")
    
    # Initialize the YOLO model with the respective weights
    selected_model = YOLO(f"{model_path}{model_type}.pt")
    
    # Perform hyperparameter tuning
    selected_model.tune(
        data=data_yaml_path,  # Path to the data.yaml file
        epochs=30,            # Number of epochs for tuning
        iterations=300,       # Number of iterations for tuning
        optimizer="AdamW",   # Optimizer to use during tuning
        plots=False,          # Skip plotting for faster tuning
        save=False,           # Skip saving checkpoints during tuning
        val=False             # Skip validation except on final epoch for faster tuning
    )
    
    # After tuning, train the model with the best hyperparameters found
    print(f"Training model: {model_type} with best hyperparameters found during tuning")
    selected_model.train(
        data=data_yaml_path,  # Path to the data.yaml file
        epochs=100,           # Set a high number of epochs
        imgsz=640,           # Image size
        batch=16,            # Batch size (adjust to match report)
        name=f"{model_type}_tuned_training",  # Name of the training run
        resume=False,        # Ensure training starts fresh
        patience=20          # Stop if no progress over 20 epochs
    )
    
    # Evaluate the model on the test set
    print(f"Evaluating model: {model_type}")
    results = selected_model.val(data=data_yaml_path)
    print(f"Test results for {model_type}:", results)
    
    # Extract mAP@0.5 and inference time for reporting
    mAP_50 = results['metrics/mAP_0.5'] if 'metrics/mAP_0.5' in results else 'N/A'
    inference_time = results['speed/inference'] if 'speed/inference' in results else 'N/A'
    results_summary.append((model_type, mAP_50, inference_time))
    
    # Save the model with a unique name
    saved_model_path = f"{model_path}{model_type}_tuned_trained.pt"
    selected_model.save(saved_model_path)
    print(f"Model saved as: {saved_model_path}")
    
    # Free up resources
    del selected_model
    torch.cuda.empty_cache()

print("All models trained and saved successfully.")

# Display summary of results
print("\nSummary of Results:")
print(f"{'Model Type':<10} | {'mAP@0.5':<10} | {'Inference Time (ms)':<20}")
print("-" * 50)
for model_type, mAP_50, inference_time in results_summary:
    print(f"{model_type:<10} | {mAP_50:<10} | {inference_time:<20}")

# Output the results to a CSV file
csv_file_path = "model_results_summary.csv"
with open(csv_file_path, mode='w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['Model Type', 'mAP@0.5', 'Inference Time (ms)'])
    for model_type, mAP_50, inference_time in results_summary:
        csv_writer.writerow([model_type, mAP_50, inference_time])

print(f"Results summary saved to {csv_file_path}")
