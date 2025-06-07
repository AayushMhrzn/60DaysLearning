import cv2

def nothing(x):
    pass  # Dummy function for trackbar callback

def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Cannot open camera.")
        return

    # Create a window and add trackbars
    cv2.namedWindow("Edge Detection Controls")
    cv2.createTrackbar("Min Threshold", "Edge Detection Controls", 50, 255, nothing)
    cv2.createTrackbar("Max Threshold", "Edge Detection Controls", 150, 255, nothing)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame.")
            break

        frame = cv2.resize(frame, (640, 480))

        # Get values from trackbars
        min_val = cv2.getTrackbarPos("Min Threshold", "Edge Detection Controls")
        max_val = cv2.getTrackbarPos("Max Threshold", "Edge Detection Controls")

        # Preprocessing
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 1)

        # Canny edge detection with dynamic thresholds
        edges = cv2.Canny(blurred, min_val, max_val)

        # Display original and edges side-by-side
        combined = cv2.hconcat([frame, cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)])
        cv2.imshow("Original | Edges", combined)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
