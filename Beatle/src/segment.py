import os
import torch
import cv2
import numpy as np

# Load the SAM (Segment Anything Model) model
def load_model(model_path):
    """Load the pre-trained SAM model from the given path."""
    model = torch.load(model_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    model.eval()  # Set the model to evaluation mode
    return model

def segment_image(model, image):
    """Perform segmentation on a single image using the SAM model."""
    # Preprocess the image as required by the SAM model
    image_tensor = torch.from_numpy(image).permute(2, 0, 1).float().unsqueeze(0)  # Convert to tensor and add batch dimension
    
    # If your model requires specific normalization, add that step here
    
    with torch.no_grad():
        output = model(image_tensor)  # Run inference with the model

    # Assuming the model returns a segmentation mask (update this according to your model's output)
    mask = output['masks'][0]  # Extract mask (adjust if needed based on the model)
    
    # Convert mask to a numpy array if needed for saving
    mask_np = mask.cpu().numpy() if torch.is_tensor(mask) else mask
    
    return mask_np

def process_images(input_dir, output_dir, model):
    """Process each image in the input directory and segment it."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.endswith(".png"):  # Assuming the images are PNG files
            image_path = os.path.join(input_dir, filename)
            print(f"Segmenting {filename}...")

            # Load the image
            image = cv2.imread(image_path)

            # Perform segmentation
            mask = segment_image(model, image)

            # Save the mask or segmented output
            mask_output_path = os.path.join(output_dir, f"segmented_{filename}")
            
            # Save the mask as a PNG (you can change this depending on how you want to save the output)
            cv2.imwrite(mask_output_path, mask * 255)  # Scale mask to [0, 255] for saving

            print(f"Saved segmented image: {mask_output_path}")

if __name__ == "__main__":
    input_folder = "/Beatle/in"   # Path to the input images (those processed from videos)
    output_folder = "/Beatle/out" # Path to save segmented images
    model_path = "/Beatle/models/sam_b.pt"  # Path to the pre-trained SAM model

    # Load the SAM model
    model = load_model(model_path)

    # Process and segment images
    process_images(input_folder, output_folder, model)
