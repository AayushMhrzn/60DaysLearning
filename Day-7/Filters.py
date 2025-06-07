import cv2
import numpy as np

# --- Filter Kernels ---
kernels = {
    "blur": np.ones((5, 5), np.float32) / 25,
    "sharpen": np.array([[0, -1, 0],
                          [-1, 5, -1],
                          [0, -1, 0]]),
    "emboss": np.array([[-2, -1, 0],
                         [-1, 1, 1],
                         [0, 1, 2]]),
    "edge_enhance": np.array([[-1, -1, -1],
                               [-1, 9, -1],
                               [-1, -1, -1]])
}

# --- Trackbar Callback (dummy) ---
def nothing(x):
    pass

cv2.namedWindow("Filtered")
cv2.createTrackbar("Filter", "Filtered", 0, len(kernels)-1, nothing)

filter_names = list(kernels.keys())

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    filter_index = cv2.getTrackbarPos("Filter", "Filtered")
    filter_name = filter_names[filter_index]
    kernel = kernels[filter_name]

    filtered = cv2.filter2D(frame, -1, kernel)

    cv2.putText(filtered, f"Filter: {filter_name}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Filtered", filtered)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
