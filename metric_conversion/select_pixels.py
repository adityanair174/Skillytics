import cv2
import numpy as np
import os

# Configuration
IMAGE_PATH = r'C:\Users\udits\data\cone_marker\original_image.jpg'
OUTPUT_FILE = 'image_pixel_coords.npy'
WINDOW_NAME = "Point_Selection_Interface" # Use one constant name

# Global state
img_display = None
clicked_points = []

def mouse_callback(event, x, y, flags, param):
    global img_display, clicked_points

    if event == cv2.EVENT_LBUTTONDOWN:
        # Store the point
        clicked_points.append((x, y))
        print(f"Captured Point {len(clicked_points)}: ({x}, {y})")

        # Draw on the display image
        # We draw a circle and a label for clarity
        cv2.circle(img_display, (x, y), 6, (0, 0, 255), -1)
        cv2.putText(img_display, str(len(clicked_points)), (x + 10, y + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

def main():
    global img_display

    if not os.path.exists(IMAGE_PATH):
        print(f"Error: File not found at {IMAGE_PATH}")
        return

    original_img = cv2.imread(IMAGE_PATH)
    if original_img is None:
        print("Error: Could not load image.")
        return

    # Create the working copy for display
    img_display = original_img.copy()

    # Setup Window
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL) # Allows resizing if image is huge
    cv2.setMouseCallback(WINDOW_NAME, mouse_callback)

    print("-" * 30)
    print("INSTRUCTIONS:")
    print("1. Click on cones to mark them.")
    print("2. Press 'c' to clear all points if you mess up.")
    print("3. Press 'q' to save and exit.")
    print("-" * 30)

    while True:
        cv2.imshow(WINDOW_NAME, img_display)
        
        # waitKey(1) allows the UI to refresh and process mouse events
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q') or key == 27: # 'q' or Esc to quit
            break
        elif key == ord('c'): # 'c' to clear and restart
            img_display = original_img.copy()
            clicked_points.clear()
            print("Points cleared. Start over.")

    cv2.destroyAllWindows()

    if clicked_points:
        points_array = np.array(clicked_points, dtype=np.float32)
        np.save(OUTPUT_FILE, points_array)
        print(f"\nSaved {len(points_array)} points to {OUTPUT_FILE}")
        print(points_array)
    else:
        print("No points selected.")

if __name__ == "__main__":
    main()