import cv2
import matplotlib.pyplot as plt

# Load image
img = cv2.imread('cat1.jpg')

# Convert to other color spaces
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

# Set up color spaces and channel labels
color_spaces = {
    'BGR': (img, ['b', 'g', 'r']),
    'HSV': (img_hsv, ['h', 's', 'v']),
    'LAB': (img_lab, ['l', 'a', 'b']),
}

# Create a single figure for all subplots
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 4))

for ax, (space_name, (image, channels)) in zip(axes, color_spaces.items()):
    for i, label in enumerate(channels):
        hist = cv2.calcHist([image], [i], None, [256], [0, 256])
        ax.plot(hist, label=label)
    ax.set_title(f'{space_name} Histogram')
    ax.legend()

plt.tight_layout()
plt.show()
