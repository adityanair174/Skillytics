import cv2
import numpy as np
import os

def animate_mapping():
    # 1. Configuration & Loading
    image_path = r'C:\Users\udits\data\cone_marker\original_image.jpg'
    matrix_path = 'homography_matrix.npy'
    video_output = 'coordinate_transformation_demo.mp4'
    
    if not os.path.exists(matrix_path):
        print("Error: homography_matrix.npy not found.")
        return

    img = cv2.imread(image_path)
    h_matrix = np.load(matrix_path)
    
    # Define metric view parameters (matching your previous script)
    scale = 50 
    w_m, h_m = 25, 35 # Adjusted to cover more points from your list
    output_size = (w_m * scale, h_m * scale)
    
    # Scaling matrix for metric display
    S = np.array([[scale, 0, 0], [0, scale, 0], [0, 0, 1]], dtype=np.float32)
    total_h = np.dot(S, h_matrix)
    
    # Generate the base top view
    top_view_base = cv2.warpPerspective(img, total_h, output_size)

    # 2. Define Animation Path (moving between key cones)
    # We'll simulate a player running through points 4 -> 10 -> 14 -> 19
    path_points = [
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
    ]
    
    # 3. Video Writer Setup
    # Combining original and top view side-by-side
    combined_w = 1280 + output_size[0]
    combined_h = max(720, output_size[1])
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_output, fourcc, 30.0, (combined_w, combined_h))

    print(f"Generating animation: {video_output}...")

    # 4. Animation Loop
    steps_per_segment = 60
    h_inv = np.linalg.inv(h_matrix) # To find pixel location from metric for the path

    for i in range(len(path_points) - 1):
        start_m = np.array(path_points[i])
        end_m = np.array(path_points[i+1])
        
        for t in np.linspace(0, 1, steps_per_segment):
            # Current Metric Position (linear interpolation)
            curr_m = (1 - t) * start_m + t * end_m
            
            # Convert Metric -> Pixel space for the original image display
            # P_pixel = H_inv * P_metric
            m_vec = np.array([curr_m[0], curr_m[1], 1.0])
            p_vec = np.dot(h_inv, m_vec)
            p_vec /= p_vec[2]
            curr_px = (int(p_vec[0]), int(p_vec[1]))
            
            # Convert Metric -> Top View Pixels
            curr_top_px = (int(curr_m[0] * scale), int(curr_m[1] * scale))

            # Draw on frames
            frame_orig = cv2.resize(img.copy(), (1280, 720))
            # Scale px for resized display
            resize_factor_x = 1280 / img.shape[1]
            resize_factor_y = 720 / img.shape[0]
            cv2.circle(frame_orig, (int(curr_px[0]*resize_factor_x), int(curr_px[1]*resize_factor_y)), 10, (0, 255, 0), -1)
            cv2.putText(frame_orig, f"Pixel Coordinate: {curr_px}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)

            frame_top = top_view_base.copy()
            cv2.circle(frame_top, curr_top_px, 12, (0, 0, 255), -1)
            cv2.putText(frame_top, f"Metric Coordinate: {curr_m[0]:.1f}m, {curr_m[1]:.1f}m", (20, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)

            # Combine side-by-side
            combined = np.zeros((combined_h, combined_w, 3), dtype=np.uint8)
            combined[:720, :1280] = frame_orig
            combined[:output_size[1], 1280:] = frame_top
            
            out.write(combined)
            cv2.imshow("Tracking Conversion Demo", combined)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    out.release()
    cv2.destroyAllWindows()
    print("Done! Video saved.")

if __name__ == "__main__":
    animate_mapping()