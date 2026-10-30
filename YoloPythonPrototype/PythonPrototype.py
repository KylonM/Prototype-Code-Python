import cv2
from threading import Thread
from ultralytics import YOLO

class CameraStream:
    """Helper class to continuously read camera frames in a background thread."""
    def __init__(self, camera_index):
        self.cap = cv2.VideoCapture(camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        self.ret, self.frame = self.cap.read()
        self.stopped = False
        self.thread = Thread(target=self.update, args=(), daemon=True)

    def start(self):
        self.thread.start()
        return self

    def update(self):
        while not self.stopped:
            if not self.cap.isOpened():
                self.stopped = True
                break
            self.ret, self.frame = self.cap.read()

    def read(self):
        return self.ret, self.frame

    def release(self):
        self.stopped = True
        self.thread.join()
        self.cap.release()

if __name__ == "__main__":
    # Get user input for number of cameras
    while True:
        try:
            num_cameras = int(input("Enter the number of cameras to run (1, 2, or 3): "))
            if 1 <= num_cameras <= 3:
                break
            print("Please enter a number between 1 and 3.")
        except ValueError:
            print("Invalid input. Please enter 1, 2, or 3.")

    # Your trained model path
    MODEL_WEIGHTS = "C:/Users/kmort/OneDrive/Documents/School/Capstone/LegoLearningModel/my_model/best.pt"
    
    # Load model on the main thread
    model = YOLO(MODEL_WEIGHTS)

    # Initialize the requested number of camera streams dynamically
    streams = []
    for i in range(num_cameras):
        stream = CameraStream(i).start()
        streams.append((i, stream))

    print(f"Starting stream with {num_cameras} camera(s) and location readouts. Press 'q' to exit.")

    try:
        while True:
            for cam_idx, stream in streams:
                ret, frame = stream.read()
                if ret:
                    res = model(frame, conf=0.5, verbose=False)
                    annotated_frame = res[0].plot()

                    # Extract and print object location data
                    boxes = res[0].boxes
                    if len(boxes) > 0:
                        print(f"\n--- Camera {cam_idx} Detections ---")
                        for box in boxes:
                            # Bounding box coordinates (xmin, ymin, xmax, ymax)
                            coords = box.xyxy[0].tolist()
                            xmin, ymin, xmax, ymax = [round(c, 1) for c in coords]
                            
                            # Center coordinates
                            x_center = round((xmin + xmax) / 2, 1)
                            y_center = round((ymin + ymax) / 2, 1)
                            
                            # Class name and confidence
                            cls_id = int(box.cls[0])
                            conf = round(float(box.conf[0]), 2)
                            class_name = model.names[cls_id]
                            
                            print(f"  Class: {class_name} | Conf: {conf} | Box: [{xmin}, {ymin}, {xmax}, {ymax}] | Center: ({x_center}, {y_center})")

                    cv2.imshow(f"YOLO11s - Camera {cam_idx}", annotated_frame)

            # Break loop on pressing 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        # Clean up all active resources safely
        for _, stream in streams:
            stream.release()
        cv2.destroyAllWindows()