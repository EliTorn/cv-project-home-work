import cv2
import numpy as np

# Read the image
image = cv2.imread('image_text.png')

# Create a white background image with the same dimensions as the original image
white_background = np.full_like(image, (255, 255, 255))

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply thresholding to segment text from background
_, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)

# Invert the binary image
binary = cv2.bitwise_not(binary)

# Find contours
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw contours on the white background
result = white_background.copy()
cv2.drawContours(result, contours, -1, (0, 0, 0), 2)  # Draw contours in black color on white background

# Display the result
cv2.imshow('Segmented Text', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
