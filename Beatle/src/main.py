import os
import torch
from ultralytics import YOLO

# Define the path to the models directory
model_path = "../models/"

# Define the list of YOLO segmentation model types
model_types = ['yolov5nu', 'yolov5su', 'yolov5mu', 'yolov5lu', 'yolov5xu','yolov8n-seg', 'yolov8s-seg', 'yolov8m-seg', 'yolov8l-seg', 'yolov8x-seg','yolo11n-seg', 'yolo11s-seg', 'yolo11m-seg', 'yolo11l-seg', 'yolo11x-seg']

# Define the list of batch sizes
batch_sizes = [8, 12]

# Iterate through each model type
for model_type in model_types:
    for batch_size in batch_sizes:
        print(f"Tuning and training model: {model_type} with batch size: {batch_size}")

        try:
            # Initialize the YOLO model with the respective weights
            selected_model = YOLO(f"{model_path}{model_type}.pt")

            # Dynamically set the path to the dataset YAML file based on the model version
            if 'yolov5' in model_type:
                data_yaml_path = "/home/chris/Documents/Github/STARRGazer/Beatle/src/annotated/v5/data.yaml"
            elif 'yolov8' in model_type:
                data_yaml_path = "/home/chris/Documents/Github/STARRGazer/Beatle/src/annotated/v8/data.yaml"
            elif 'yolo11' in model_type:
                data_yaml_path = "/home/chris/Documents/Github/STARRGazer/Beatle/src/annotated/v11/data.yaml"
            else:
                raise ValueError(f"Unknown model type: {model_type}")

            # Train the model with the current batch size
            print(f"Training model: {model_type} with batch size: {batch_size}")
            selected_model.train(
                data=data_yaml_path,                                    # Path to the data.yaml file
                epochs=200,                                             # Set a high number of epochs so hopefully patience is reached first
                imgsz=640,                                              # Image size 
                batch=batch_size,                                       # Current batch size
                name=f"{model_type}_batch{batch_size}",                 # Unique name to batch size and model 
                patience=10,                                            # Stop if no progress over 10 epochs
                plots=False,                                            # Dont plot f curves etc in final validation
                cache=False,                                            # Fix memory problems 
                optimizer="SGD"
            )

            # Save the model with a unique name
            saved_model_path = f"{model_path}{model_type}_batch{batch_size}.pt"
            selected_model.save(saved_model_path)
            print(f"Model saved as: {saved_model_path}")

            # Free up resources
            del selected_model
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"Error encountered with model {model_type} and batch size {batch_size}: {e}")
            continue

print("All models trained and evaluated successfully.")
