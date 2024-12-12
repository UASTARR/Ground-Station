import cv2
import torch
from ultralytics import YOLO
import os

def segment_video(video_path, model_path, output_path):
    # Check if the video file exists
    if not os.path.exists(video_path):
        print(f"Video file not found: {video_path}")
        return

    # Check if the model file exists
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        return

    try:
        # Load the YOLO model
        model = YOLO(model_path)

        # Open the video file
        video = cv2.VideoCapture(video_path)
        if not video.isOpened():
            print(f"Error opening video file: {video_path}")
            return

        # Get video properties
        frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(video.get(cv2.CAP_PROP_FPS))
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

        # Define the codec and create VideoWriter object
        output_video = cv2.VideoWriter(
            output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height)
        )

        print(f"Processing video: {video_path}")

        frame_count = 0
        while True:
            ret, frame = video.read()
            if not ret:
                break

            # Run segmentation on the frame
            results = model(frame)

            # Extract segmented frame
            segmented_frame = results[0].plot()

            # Write the segmented frame to the output video
            output_video.write(segmented_frame)

            frame_count += 1
            print(f"Processed frame {frame_count}/{total_frames}", end='\r')

        # Release resources
        video.release()
        output_video.release()
        print(f"\nSegmentation completed. Output saved to: {output_path}")

    except Exception as e:
        print(f"Error during video segmentation: {e}")

# Example usage
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run video segmentation using a YOLO model.")
    parser.add_argument("video_path", type=str, help="Path to the input video file.")
    parser.add_argument("model_path", type=str, help="Path to the YOLO model file.")
    parser.add_argument("output_path", type=str, help="Path to save the output segmented video.")

    args = parser.parse_args()

    segment_video(args.video_path, args.model_path, args.output_path)
