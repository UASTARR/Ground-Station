import os
import cv2
import shutil

def clear_output_dir(output_dir):
    """Clears the output directory if it exists."""
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)  # Remove the directory and all its contents
    os.makedirs(output_dir)  # Recreate the empty directory

def process_videos(input_dir, output_dir):
    # Clear the output directory before saving new frames
    clear_output_dir(output_dir)

    # Iterate over all files in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith(".mov") or filename.endswith(".mp4"):  # Adjust for other video formats as needed
            video_path = os.path.join(input_dir, filename)
            print(f"Processing {filename}...")

            # Capture the video
            vid_cap = cv2.VideoCapture(video_path)
            count = 0

            while True:
                success, image = vid_cap.read()

                # Break the loop if no more frames are available
                if not success:
                    break

                # Save the frame as a PNG file
                frame_filename = f"{os.path.splitext(filename)[0]}_frame_{count}.png"
                frame_output_path = os.path.join(output_dir, frame_filename)
                
                # Save the image frame as a PNG
                cv2.imwrite(frame_output_path, image)
                
                # Move to the next frame
                count += 1

            print(f"Finished processing {filename}. Total frames: {count}")

if __name__ == "__main__":
    input_folder = "Beatle/db"   # Adjust to your actual path to 'db'
    output_folder = "Beatle/in"  # Adjust to your actual path to 'in'

    process_videos(input_folder, output_folder)
