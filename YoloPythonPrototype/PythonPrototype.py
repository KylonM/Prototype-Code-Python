import cv2
from threading import Thread
from ultralytics import YOLO
import numpy as np
from localization import (
    localize_object,
    get_camera_position,
    get_camera_orientation
)

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
    MODEL_WEIGHTS = "best.pt"
    
    # Load model on the main thread
    model = YOLO(MODEL_WEIGHTS)
    print("Model classes:", model.names)

    # Initialize the requested number of camera streams dynamically

    # Temporary camera calibration values
    # These will be replaced with measured values later
    K_laptop = np.array([
        [500, 0, 320],
        [0, 500, 240],
        [0, 0, 1]
    ])

    K_iphone = np.array([
        [500, 0, 320],
        [0, 500, 240],
        [0, 0, 1]
    ])

    # Camera positions and orientations
    camera1 = {
        "position": get_camera_position("Laptop"),
        "R": get_camera_orientation("Laptop"),
        "K": K_laptop
    }

    camera2 = {
        "position": get_camera_position("iPhone"),
        "R": get_camera_orientation("iPhone"),
        "K": K_iphone
    }
    # Create a display for the calculated XYZ coordinates
    location_display = np.zeros((240, 480, 3), dtype=np.uint8)
    streams = []
    for i in range(num_cameras):
        stream = CameraStream(i).start()
        streams.append((i, stream))

    print(f"Starting stream with {num_cameras} camera(s) and location readouts. Press 'q' to exit.")
    printed_resolutions = set()
    try:
        while True:
            detections = {}
            for cam_idx, stream in streams:
                ret, frame = stream.read()
                if ret:

                    if cam_idx not in printed_resolutions:
                        height, width = frame.shape[:2]
                        print(f"Camera {cam_idx} resolution: {width} x {height}")
                        printed_resolutions.add(cam_idx)
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
                            detections.setdefault(cam_idx, []).append({
                                "class": class_name,
                                "center": (x_center, y_center),
                                "confidence": conf
                            })
                            
                            print(f"  Class: {class_name} | Conf: {conf} | Box: [{xmin}, {ymin}, {xmax}, {ymax}] | Center: ({x_center}, {y_center})")

                    cv2.imshow(f"Drone Detection - Camera {cam_idx}", annotated_frame)

            # Break loop on pressing 'q'

            # Triangulate only an unambiguous detection pair
            if 0 in detections and 1 in detections:
                objects0 = detections[0]
                objects1 = detections[1]

                persons0 = [obj for obj in objects0 if obj["class"] == "drone"]
                persons1 = [obj for obj in objects1 if obj["class"] == "drone"]

                if len(persons0) == 1 and len(persons1) == 1:
                    obj0 = persons0[0]
                    obj1 = persons1[0]

                    if obj0["class"] == obj1["class"]:

                        try:
                            position, error = localize_object(
                                obj0["center"],
                                obj1["center"],
                                camera1,
                                camera2
                            )

                            # Update the localization display
                            location_display[:] = (30, 30, 30)

                            cv2.putText(
                                location_display,
                                "STEREO LOCALIZATION",
                                (35, 40),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.8, (255, 255, 255), 2
                            )

                            cv2.putText(
                                location_display,
                                f"X: {position[0]:.2f} ft",
                                (35, 90),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.8, (0, 255, 0), 2
                            )

                            cv2.putText(
                                location_display,
                                f"Y: {position[1]:.2f} ft",
                                (35, 130),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.8, (0, 255, 0), 2
                            )

                            cv2.putText(
                                location_display,
                                f"Z: {position[2]:.2f} ft",
                                (35, 170),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.8, (0, 255, 0), 2
                            )

                            cv2.putText(
                                location_display,
                                f"Ray gap: {error:.3f} ft",
                                (35, 210),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (255, 255, 255), 1
                            )

                            print(
                                f"Estimated XYZ: {position} ft | "
                                f"Ray gap: {error:.4f} ft"
                            )

                        except ValueError as e:
                            print(f"Triangulation skipped: {e}")

            cv2.imshow("Stereo Localization", location_display)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        # Clean up all active resources safely
        for _, stream in streams:
            stream.release()
        cv2.destroyAllWindows()