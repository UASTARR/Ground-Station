from ultralytics import SAM, YOLO

# Define the path to the models directory
model_path = "Beatle/models/"

# Load the SAM model
sam_model = SAM(f"{model_path}sam_b.pt")

# Load YOLO models from the specified path
yolo11n_model = YOLO(f"{model_path}yolo11n.pt")
yolo11s_model = YOLO(f"{model_path}yolo11s.pt")
yolo11m_model = YOLO(f"{model_path}yolo11m.pt")
yolo11l_model = YOLO(f"{model_path}yolo11l.pt")
yolo11x_model = YOLO(f"{model_path}yolo11x.pt")

# Display model information (optional)
sam_model.info()
yolo11n_model.info()
yolo11s_model.info()
yolo11m_model.info()
yolo11l_model.info()
yolo11x_model.info()
