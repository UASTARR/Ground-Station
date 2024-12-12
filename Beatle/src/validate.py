import os
import csv
from ultralytics import YOLO

# Define paths and parameters
model_path = "../models/"
data_paths = {
    "yolov5": "/home/chris/Documents/Github/STARRGazer/Beatle/src/annotated/v5/data.yaml",
    "yolov8": "/home/chris/Documents/Github/STARRGazer/Beatle/src/annotated/v8/data.yaml",
    "yolo11": "/home/chris/Documents/Github/STARRGazer/Beatle/src/annotated/v11/data.yaml"
}

confidence_levels = [0.7, 0.8, 0.9]

# Initialize results summary
results_summary = []

# Define model types and batch sizes (should match the training script)
model_types = [
    'yolov5nu', 'yolov5su', 'yolov5mu', 'yolov5lu', 'yolov5xu',
    'yolov8n-seg', 'yolov8s-seg', 'yolov8m-seg', 'yolov8l-seg', 'yolov8x-seg',
    'yolo11n-seg', 'yolo11s-seg', 'yolo11m-seg', 'yolo11l-seg', 'yolo11x-seg'
]

batch_sizes = [8, 12]

# Loop through each model and batch size
for model_type in model_types:
    for batch_size in batch_sizes:
        model_file = f"{model_path}{model_type}_batch{batch_size}.pt"

        if not os.path.exists(model_file):
            print(f"Model file not found: {model_file}. Skipping.")
            continue

        try:
            # Determine the correct data YAML path
            if 'yolov5' in model_type:
                data_yaml_path = data_paths["yolov5"]
            elif 'yolov8' in model_type:
                data_yaml_path = data_paths["yolov8"]
            elif 'yolo11' in model_type:
                data_yaml_path = data_paths["yolo11"]
            else:
                print(f"Unknown model type: {model_type}. Skipping.")
                continue

            # Load the trained model
            selected_model = YOLO(model_file)

            # Evaluate the model at different confidence levels
            for conf in confidence_levels:
                print(f"Evaluating model: {model_type} with batch size: {batch_size} and confidence: {conf}")
                results = selected_model.val(data=data_yaml_path, conf=conf, split="test")

                # Extract relevant metrics
                mAP_50 = results.box.map50
                inference_time = results.speed.get("inference")
                results_summary.append((model_type, batch_size, conf, mAP_50, inference_time))

        except Exception as e:
            print(f"Error during evaluation for model {model_type}, batch size {batch_size}: {str(e)}")
            continue

# Display summary of results
print("\nSummary of Results:")
print(f"{'Model Type':<20} | {'Batch Size':<10} | {'Confidence':<10} | {'mAP@0.5':<10} | {'Inference Time (ms)':<20}")
print("-" * 80)
for model_type, batch_size, conf, mAP_50, inference_time in results_summary:
    print(f"{model_type:<20} | {batch_size:<10} | {conf:<10} | {mAP_50:<10} | {inference_time:<20}")

# Output the results to a CSV file
csv_file_path = "model_results_summary.csv"
with open(csv_file_path, mode='w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['Model Type', 'Batch Size', 'Confidence', 'mAP@0.5', 'Inference Time (ms)'])
    for model_type, batch_size, conf, mAP_50, inference_time in results_summary:
        csv_writer.writerow([model_type, batch_size, conf, mAP_50, inference_time])

print(f"Results summary saved to {csv_file_path}")
