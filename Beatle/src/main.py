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

# Define the list of YOLO segmentation model types
model_types = [
    'yolov5n-seg', 'yolov5s-seg', 'yolov5m-seg', 'yolov5l-seg', 'yolov5x-seg',
    'yolov8n-seg', 'yolov8s-seg', 'yolov8m-seg', 'yolov8l-seg', 'yolov8x-seg',
    'yolo11n-seg', 'yolo11s-seg', 'yolo11m-seg', 'yolo11l-seg', 'yolo11x-seg'
]

# Define the list of batch sizes
batch_sizes = [1, 2, 4]

# Define the confidence levels for evaluation
confidence_levels = [0.7, 0.9]

# Results summary
results_summary = []

# Iterate through each model type
for model_type in model_types:
    for batch_size in batch_sizes:
        print(f"Tuning and training model: {model_type} with batch size: {batch_size}")

        # Initialize the YOLO model with the respective weights
        selected_model = YOLO(f"{model_path}{model_type}.pt")

        # Train the model with the current batch size
        print(f"Training model: {model_type} with batch size: {batch_size}")
        selected_model.train(
            data=data_yaml_path,  # Path to the data.yaml file
            epochs=100,           # Set a high number of epochs
            imgsz=640,           # Image size
            batch=batch_size,    # Current batch size
            name=f"{model_type}_batch{batch_size}_tuned_training",  # Unique name
            resume=False,        # Ensure training starts fresh
            patience=20,          # Stop if no progress over 20 epochs
            optimizer="SGD"
        )

        # Evaluate the model with different confidence levels
        for conf in confidence_levels:
            print(f"Evaluating model: {model_type} with batch size: {batch_size} and confidence: {conf}")
            results = selected_model.val(data=data_yaml_path, conf=conf)
            print(f"Test results for {model_type} (Batch size: {batch_size}, Confidence: {conf}):", results)

            # Extract mAP@0.5 and inference time for reporting
            mAP_50 = results['metrics/mAP_0.5'] if 'metrics/mAP_0.5' in results else 'N/A'
            inference_time = results['speed/inference'] if 'speed/inference' in results else 'N/A'
            results_summary.append((model_type, batch_size, conf, mAP_50, inference_time))

        # Save the model with a unique name
        saved_model_path = f"{model_path}{model_type}_batch{batch_size}_tuned_trained.pt"
        selected_model.save(saved_model_path)
        print(f"Model saved as: {saved_model_path}")

        # Free up resources
        torch.cuda.empty_cache()

print("All models trained and evaluated successfully.")

# Display summary of results
print("\nSummary of Results:")
print(f"{'Model Type':<10} | {'Batch Size':<10} | {'Confidence':<10} | {'mAP@0.5':<10} | {'Inference Time (ms)':<20}")
print("-" * 70)
for model_type, batch_size, conf, mAP_50, inference_time in results_summary:
    print(f"{model_type:<10} | {batch_size:<10} | {conf:<10} | {mAP_50:<10} | {inference_time:<20}")

# Output the results to a CSV file
csv_file_path = "model_results_summary.csv"
with open(csv_file_path, mode='w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['Model Type', 'Batch Size', 'Confidence', 'mAP@0.5', 'Inference Time (ms)'])
    for model_type, batch_size, conf, mAP_50, inference_time in results_summary:
        csv_writer.writerow([model_type, batch_size, conf, mAP_50, inference_time])

print(f"Results summary saved to {csv_file_path}")
