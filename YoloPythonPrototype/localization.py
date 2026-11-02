
import numpy as np

def pixel_to_ray(u, v, K):
    """
    Convert an object's pixel coordinates into
    a 3D viewing direction in camera coordinates.
    """
    pixel = np.array([u, v, 1.0])

    ray = np.linalg.inv(K) @ pixel

    ray = ray / np.linalg.norm(ray)

    return ray

def camera_to_world(ray_camera, R):
    """
    Convert a viewing direction from camera coordinates
    into world coordinates.

    R is the camera-to-world rotation matrix.
    """
    ray_world = R @ ray_camera

    return ray_world

def create_viewing_ray(u, v, K, R, camera_position):
    """
    Create a 3D viewing ray from a camera.

    u, v: Detected object's pixel coordinates
    K: Camera intrinsic matrix
    R: Camera-to-world rotation matrix
    camera_position: Camera XYZ position in world coordinates

    Returns:
        origin: Camera position
        direction: Unit viewing direction in world coordinates
    """
    ray_camera = pixel_to_ray(u, v, K)

    ray_world = camera_to_world(ray_camera, R)

    origin = np.asarray(camera_position, dtype=float)

    return origin, ray_world


def triangulate_rays(origin1, direction1, origin2, direction2):
    """
    Estimate a 3D object position from two viewing rays.

    Returns:
        position: Estimated XYZ object coordinates
        error: Distance between the closest points on the rays
    """
    C1 = np.asarray(origin1, dtype=float)
    C2 = np.asarray(origin2, dtype=float)

    d1 = np.asarray(direction1, dtype=float)
    d2 = np.asarray(direction2, dtype=float)

    d1 = d1 / np.linalg.norm(d1)
    d2 = d2 / np.linalg.norm(d2)

    # Find the closest points on the two viewing lines
    A = np.column_stack((d1, -d2))
    b = C2 - C1

    t1, t2 = np.linalg.lstsq(A, b, rcond=None)[0]

    # Reject intersections behind either camera
    if t1 < 0 or t2 < 0:
        raise ValueError("Triangulated point is behind a camera.")

    P1 = C1 + t1 * d1
    P2 = C2 + t2 * d2

    # Estimate the object position as the midpoint
    position = (P1 + P2) / 2

    # Measure how closely the rays agree
    error = np.linalg.norm(P1 - P2)

    return position, error


def localize_object(pixel1, pixel2, camera1, camera2):
    """
    Calculate an object's XYZ position using two cameras.
    """

    origin1, direction1 = create_viewing_ray(
        pixel1[0], pixel1[1],
        camera1["K"], camera1["R"], camera1["position"]
    )

    origin2, direction2 = create_viewing_ray(
        pixel2[0], pixel2[1],
        camera2["K"], camera2["R"], camera2["position"]
    )

    return triangulate_rays(
        origin1, direction1, origin2, direction2
    )

def get_camera_position(camera_name):
    """
    Ask the user to enter a camera's XYZ position.
    All distances must use the same units.
    """
    print(f"\nEnter position for {camera_name}:")

    x = float(input("X position (ft): "))
    y = float(input("Y position (ft): "))
    z = float(input("Z position (ft): "))

    return np.array([x, y, z])

def rotation_from_angles(yaw, pitch, roll):
    """
    Camera-to-world rotation matrix.

    Angles in degrees:
        Positive yaw: right
        Positive pitch: up
        Positive roll: clockwise looking forward

    World axes: X right, Y forward, Z up.
    """

    yaw, pitch, roll = np.radians([yaw, pitch, roll])

    # OpenCV camera axes -> world axes at zero orientation
    R_base = np.array([
        [1,  0, 0],
        [0,  0, 1],
        [0, -1, 0]
    ])

    # Yaw: rotation about world Z
    R_yaw = np.array([
        [ np.cos(yaw), -np.sin(yaw), 0],
        [ np.sin(yaw),  np.cos(yaw), 0],
        [0, 0, 1]
    ])

    # Pitch: rotation about local camera X
    R_pitch = np.array([
        [1, 0, 0],
        [0, np.cos(pitch), -np.sin(pitch)],
        [0, np.sin(pitch),  np.cos(pitch)]
    ])

    # Roll: rotation about local camera Z (optical axis)
    R_roll = np.array([
        [np.cos(roll), -np.sin(roll), 0],
        [np.sin(roll),  np.cos(roll), 0],
        [0, 0, 1]
    ])

    return R_yaw.T @ R_base @ R_pitch.T @ R_roll


def get_camera_orientation(camera_name):
    """
    Ask the user for camera orientation in degrees.
    Returns a camera-to-world rotation matrix.
    """
    print(f"\nEnter orientation for {camera_name}:")

    yaw = float(input("Yaw (degrees, right positive): "))
    pitch = float(input("Pitch (degrees, up positive): "))
    roll = float(input("Roll (degrees, clockwise positive): "))

    return rotation_from_angles(yaw, pitch, roll)


if __name__ == "__main__":


    # Example camera calibration (not yet measured)
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

    R_laptop = get_camera_orientation("Laptop")
    R_iphone = get_camera_orientation("iPhone")

    camera1 = {
        "position": get_camera_position("Laptop"),
        "K": K_laptop,
        "R": R_laptop
    }

    camera2 = {
        "position": get_camera_position("iPhone"),
        "K": K_iphone,
        "R": R_iphone
    }


    # Simulated YOLO pixel detections
    pixel1 = (395, 240)
    pixel2 = (245, 240)

    # Calculate the object's 3D position
    position, error = localize_object(
        pixel1, pixel2, camera1, camera2
    )

    print("Estimated XYZ position:", position)
    print("Ray intersection error:", error)

