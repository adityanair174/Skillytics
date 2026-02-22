import cv2
import numpy as np
import os

def calculate_robust_homography():
    # 1. Load the pixel coordinates from your selection tool
    input_file = 'image_pixel_coords.npy'
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found. Please run the selector first.")
        return

    # Load pixel points (N, 2)
    pixel_points = np.load(input_file)
    num_points = len(pixel_points)
    
    # 2. DEFINE METRIC COORDINATES (X, Y)
    # MUST MATCH THE ORDER OF YOUR CLICKS (1, 2, 3...)
    # Based on your image markings (e.g., Point 1 is 0,0; Point 2 is 5,0 etc.)
    # Replace these placeholder values with your full list of measurements
    metric_points = np.array([
        [0.0, 0.0],   # Point 1 
        [5.0, 0.0],   # Point 2
        [10.0, 0.0],  # Point 3
        [15.0, 0.0],  # Point 4
        [20.0, 0.0],  # Point 5
        [25.0, 0.0],  # Point 6
        [0.0, 5.0],   # Point 7
        [5.0, 5.0],   # Point 8
        [10.0, 5.0],  # Point 9
        [15.0, 5.0],  # Point 10
        [20.0, 5.0],  # Point 11
        [25.0, 5.0],  # Point 12
        [0.0, 10.0],  # Point 13
        [5.0, 10.0],  # Point 14
        [10.0, 10.0], # Point 15
        [15.0, 10.0], # Point 16
        [20.0, 10.0], # Point 17
        [25.0, 10.0], # Point 18
        [5.0, 15.0],  # Point 19
        [10.0, 15.0], # Point 20
        [15.0, 15.0], # Point 21
        [20.0, 15.0], # Point 22
        [25.0, 15.0], # Point 23
        [5.0, 20.0],  # Point 24
        [10.0, 20.0], # Point 25
        [15.0, 20.0], # Point 26
        [20.0, 20.0], # Point 27
        [25.0, 20.0], # Point 28
        [5.0, 25.0],  # Point 29
        [10.0, 25.0], # Point 30
        [15.0, 25.0], # Point 31
        [20.0, 25.0], # Point 32
        [25.0, 25.0], # Point 33
        [10.0, 30.0], # Point 34
        [15.0, 30.0], # Point 35
        [20.0, 30.0], # Point 36
        [25.0, 30.0], # Point 37
        [20.0, 35.0], # Point 38
        [25.0, 35.0], # Point 39
    ], dtype=np.float32)

    # Safety check: ensure the number of points match
    if len(pixel_points) != len(metric_points):
        print(f"Mismatch: You clicked {len(pixel_points)} points but defined {len(metric_points)} metric coordinates.")
        # For this script to run, let's truncate to the smaller length for demonstration
        # min_len = min(len(pixel_points), len(metric_points))
        # pixel_points = pixel_points[:min_len]
        # metric_points = metric_points[:min_len]
        raise ValueError("Number of pixel points must match number of metric points.")

    # 3. CALCULATE HOMOGRAPHY WITH RANSAC
    # cv2.RANSAC helps ignore "bad clicks" or outliers
    # ransacReprojThreshold: Max allowed error in pixels (e.g., 5.0)
    h_matrix, mask = cv2.findHomography(pixel_points, metric_points, cv2.RANSAC, 3.0)

    if h_matrix is not None:
        print("\n--- Homography Calculation Successful ---")
        print(f"Inliers: {np.sum(mask)} / {len(mask)}")
        print("\nHomography Matrix (H):")
        print(h_matrix)
        
        # Save the matrix for use in your tracking module
        np.save('homography_matrix.npy', h_matrix)
        print("\nMatrix saved as 'homography_matrix.npy'")
    else:
        print("Homography could not be calculated.")

    return h_matrix

def transform_point(h_matrix, pixel_x, pixel_y):
    """Example function to convert a single pixel to metric space"""
    p = np.array([pixel_x, pixel_y, 1.0]).reshape(3, 1)
    world_p = np.dot(h_matrix, p)
    world_p /= world_p[2] # Normalize (z=1)
    return world_p[0][0], world_p[1][0]

def generate_top_view():
    # 1. Configuration
    image_path = r'C:\Users\udits\data\cone_marker\original_image.jpg'
    matrix_path = 'homography_matrix.npy'
    
    # Define the output image size in pixels
    # Let's say 1 meter = 50 pixels. Adjust 'scale' to zoom in/out.
    scale = 50 
    width_meters = 20  # Total width of the area you want to see
    height_meters = 30 # Total height of the area you want to see
    
    output_size = (width_meters * scale, height_meters * scale)

    # 2. Load data
    if not os.path.exists(matrix_path):
        print("Error: Homography matrix not found. Run the calculation script first.")
        return

    img = cv2.imread(image_path)
    h_matrix = np.load(matrix_path)

    # 3. Create a Scaling Matrix
    # The H matrix takes pixels -> meters (0 to 20). 
    # We need an extra step to take meters -> output pixels (0 to 1000).
    S = np.array([
        [scale, 0, 0],
        [0, scale, 0],
        [0, 0, 1]
    ], dtype=np.float32)
    
    # Combined matrix: Pixels -> Metric Space -> Scaled Output Pixels
    total_h = np.dot(S, h_matrix)

    # 4. Apply the Warp
    print("Applying perspective warp...")
    top_view = cv2.warpPerspective(img, total_h, output_size)

    # 5. Display and Save
    cv2.imshow("Original Image", cv2.resize(img, (800, 450)))
    cv2.imshow("Top View (Metric Space)", top_view)
    
    cv2.imwrite("football_court_top_view.png", top_view)
    print("Top view saved as 'football_court_top_view.png'")
    
    print("\nPress any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    calculate_robust_homography()
    generate_top_view()