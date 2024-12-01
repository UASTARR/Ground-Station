import cv2
import threading
import numpy as np
import socket
import time


class CSI_Camera:

    def __init__(self):
        # Initialize instance variables
        # OpenCV video capture element
        self.video_capture = None
        # The last captured image from the camera
        self.frame = None
        self.grabbed = False
        # The thread where the video capture runs
        self.read_thread = None
        self.read_lock = threading.Lock()
        self.running = False

    def open(self, gstreamer_pipeline_string):
        try:
            self.video_capture = cv2.VideoCapture(
                gstreamer_pipeline_string, cv2.CAP_GSTREAMER
            )
            # Grab the first frame to start the video capturing
            self.grabbed, self.frame = self.video_capture.read()

        except RuntimeError:
            self.video_capture = None
            print("Unable to open camera")
            print("Pipeline: " + gstreamer_pipeline_string)

    def start(self):
        if self.running:
            print("Video capturing is already running")
            return None
        # create a thread to read the camera image
        if self.video_capture != None:
            self.running = True
            self.read_thread = threading.Thread(target=self.updateCamera)
            self.read_thread.start()
        return self

    def stop(self):
        self.running = False
        # Kill the thread
        self.read_thread.join()
        self.read_thread = None

    def updateCamera(self):
        # This is the thread to read images from the camera
        while self.running:
            try:
                grabbed, frame = self.video_capture.read()
                with self.read_lock:
                    self.grabbed = grabbed
                    self.frame = frame
            except RuntimeError:
                print("Could not read image from camera")
        # FIX ME - stop and cleanup thread
        # Something bad happened

    def read(self):
        with self.read_lock:
            frame = self.frame.copy()
            grabbed = self.grabbed
        return grabbed, frame

    def release(self):
        if self.video_capture != None:
            self.video_capture.release()
            self.video_capture = None
        # Now kill the thread
        if self.read_thread != None:
            self.read_thread.join()


def gstreamer_pipeline(
    sensor_id=0,
    capture_width=1920,
    capture_height=1080,
    display_width=1920,
    display_height=1080,
    framerate=30,
    flip_method=0,
):
    return (
        "nvarguscamerasrc sensor-id=%d ! "
        "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            sensor_id,
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )


def stream_camera(camera, socket_conn, writer, lock, stop_flag):
    """Stream camera frames to a client socket and save video locally."""
    while not stop_flag["stop"]:
        if camera.isOpened():
            grabbed, frame = camera.read()
            if grabbed:
                # Compress frame to JPEG for streaming
                _, buffer = cv2.imencode(".jpg", frame)

                # Split into smaller chunks
                CHUNK_SIZE = 4096  # 4 KB chunks
                data = buffer.tobytes()
                data_size = len(data)

                # Stream data to the client
                try:
                    socket_conn.sendall(len(data).to_bytes(4, byteorder="big"))

                    # Send the data in chunks
                    for i in range(0, data_size, CHUNK_SIZE):
                        socket_conn.sendall(data[i : i + CHUNK_SIZE])

                except Exception as e:
                    print(f"Connection error: {e}")
                    break

                # Save the frame to video
                writer.write(frame)
        else:
            time.sleep(0.1)  # Wait if the camera is not ready


def main():
    # Camera Initialization
    left_camera = CSI_Camera()
    left_camera.open(
        gstreamer_pipeline(
            sensor_id=0,
            capture_width=1920,
            capture_height=1080,
            flip_method=0,
            display_width=960,
            display_height=540,
        )
    )
    left_camera.start()

    right_camera = CSI_Camera()
    right_camera.open(
        gstreamer_pipeline(
            sensor_id=1,
            capture_width=1920,
            capture_height=1080,
            flip_method=0,
            display_width=960,
            display_height=540,
        )
    )
    right_camera.start()

    # Network Initialization
    BUFFER_SIZE = left_camera.frame.size + right_camera.frame.size
    host = "192.168.1.101"
    port1, port2 = 3001, 3002

    try:
        # Create sockets for left and right cameras
        s1 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s1.bind((host, port1))
        s1.listen(1)

        s2 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s2.bind((host, port2))
        s2.listen(1)

        print(f"Waiting for connections on {host}:{port1} and {host}:{port2}")
        conn1, _ = s1.accept()
        conn2, _ = s2.accept()
        print("Clients connected")

    except Exception as e:
        print(f"Socket setup failed: {e}")
        return

    # Video saving
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # left_writer = cv2.VideoWriter('left_camera.avi', fourcc, 30.0, (960, 540))
    # right_writer = cv2.VideoWriter('right_camera.avi', fourcc, 30.0, (960, 540))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Use 'mp4v' for MP4 format
    left_writer = cv2.VideoWriter("left_camera.mp4", fourcc, 30.0, (960, 540))
    right_writer = cv2.VideoWriter("right_camera.mp4", fourcc, 30.0, (960, 540))

    # Thread control
    stop_flag = {"stop": False}
    lock1 = threading.Lock()
    lock2 = threading.Lock()

    # Start streaming threads
    left_thread = threading.Thread(
        target=stream_camera, args=(left_camera, conn1, left_writer, lock1, stop_flag)
    )
    right_thread = threading.Thread(
        target=stream_camera, args=(right_camera, conn2, right_writer, lock2, stop_flag)
    )
    left_thread.start()
    right_thread.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping threads...")
        stop_flag["stop"] = True

    # Clean up
    left_thread.join()
    right_thread.join()
    left_camera.stop()
    left_camera.release()
    right_camera.stop()
    right_camera.release()
    left_writer.release()
    right_writer.release()
    conn1.close()
    conn2.close()


if __name__ == "__main__":
    main()
