import cv2
import threading


# Function to get GStreamer pipeline for high-quality local recording
def gstreamer_pipeline_for_local(
    sensor_id=0,
    capture_width=1920,
    capture_height=1080,
    framerate=30,
    flip_method=0,
):
    return (
        "nvarguscamerasrc sensor-id=%d ! "
        "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink"
        % (sensor_id, capture_width, capture_height, framerate, flip_method)
    )


# Function to get GStreamer pipeline for H.264 streaming
def gstreamer_pipeline_for_streaming(
    sensor_id=0,
    host="192.168.1.101",
    port=5000,
    capture_width=1920,
    capture_height=1080,
    framerate=30,
    flip_method=0,
):
    return (
        "nvarguscamerasrc sensor-id=%d ! "
        "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, format=(string)I420 ! omxh264enc bitrate=4000000 ! "
        "rtph264pay ! udpsink host=%s port=%d"
        % (sensor_id, capture_width, capture_height, framerate, flip_method, host, port)
    )


# Thread to handle local high-resolution recording
def local_recording(sensor_id, filename, stop_flag):
    pipeline = gstreamer_pipeline_for_local(sensor_id=sensor_id)
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

    if not cap.isOpened():
        print(f"Unable to open camera {sensor_id} for local recording")
        return

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Create a high-quality video writer
    writer = cv2.VideoWriter(
        filename, cv2.VideoWriter_fourcc(*"XVID"), fps, (frame_width, frame_height)
    )
    print(f"Recording high-resolution video for camera {sensor_id}...")

    while not stop_flag["stop"]:
        ret, frame = cap.read()
        if ret:
            writer.write(frame)
        else:
            print(f"Failed to read from camera {sensor_id} during local recording")
            break

    # Release resources
    cap.release()
    writer.release()
    print(f"Stopped high-resolution recording for camera {sensor_id}")


# Thread to handle streaming
def streaming(sensor_id, stop_flag, host="192.168.1.101", port=5000):
    pipeline = gstreamer_pipeline_for_streaming(
        sensor_id=sensor_id, host=host, port=port
    )
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

    if not cap.isOpened():
        print(f"Unable to open camera {sensor_id} for streaming")
        return

    print(f"Streaming H.264 video from camera {sensor_id} to {host}:{port}...")

    while not stop_flag["stop"]:
        ret, _ = cap.read()
        if not ret:
            print(f"Failed to stream from camera {sensor_id}")
            break

    # Release resources
    cap.release()
    print(f"Stopped streaming for camera {sensor_id}")


def main():
    stop_flag = {"stop": False}  # Shared flag to stop threads
    camera_1_id = 0  # First camera ID
    camera_2_id = 1  # Second camera ID

    # Start threads for camera 1
    threading.Thread(
        target=local_recording,
        args=(camera_1_id, "camera_1_high_quality.avi", stop_flag),
    ).start()
    threading.Thread(
        target=streaming, args=(camera_1_id, stop_flag, "192.168.1.101", 5000)
    ).start()

    # Start threads for camera 2
    threading.Thread(
        target=local_recording,
        args=(camera_2_id, "camera_2_high_quality.avi", stop_flag),
    ).start()
    threading.Thread(
        target=streaming, args=(camera_2_id, stop_flag, "192.168.1.101", 5001)
    ).start()

    try:
        while True:
            # Main thread can handle other tasks or monitor the status
            pass
    except KeyboardInterrupt:
        print("Stopping all threads...")
        stop_flag["stop"] = True


if __name__ == "__main__":
    main()
