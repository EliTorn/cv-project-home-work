import cv2
import numpy as np
import matplotlib.pyplot as plt


def gradient_simple(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grad_x = np.diff(img_gray, axis=1, append=0)
    grad_y = np.diff(img_gray, axis=0, append=0)
    grad = np.sqrt(grad_x ** 2 + grad_y ** 2)
    grad_orient = np.arctan2(grad_y, grad_x) * (180 / np.pi) % 180
    return img_gray, grad_x, grad_y, grad, grad_orient


def show_gradients(img, img_gray, grad_x, grad_y, grad, grad_orient):
    fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(15, 15))
    ax[0, 0].imshow(img)
    ax[0, 0].set_title('Original Image')
    ax[0, 1].imshow(img_gray, cmap='gray')
    ax[0, 1].set_title('Gray Image')

    plt.show()


def gradient_sobel(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grad_x = cv2.Sobel(img_gray, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=3)
    grad_y = cv2.Sobel(img_gray, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=3)
    grad = cv2.magnitude(grad_x, grad_y)
    grad_orient = cv2.phase(np.array(grad_x, np.float32), np.array(grad_y, dtype=np.float32), angleInDegrees=True)
    return img_gray, grad_x, grad_y, grad, grad_orient


img_dog = cv2.imread('dog2.jpeg', cv2.IMREAD_COLOR)
imhead = img_dog.copy()

img_gray, grad_x, grad_y, grad, grad_orient = gradient_sobel(imhead)
show_gradients(imhead, img_gray, grad_x, grad_y, grad, grad_orient)


def gradient_canny(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edge = cv2.Canny(img_gray, 80, 200)
    return img_gray, edge


img_gray, edge = gradient_canny(img_dog)

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
ax[0].imshow(img_dog)
ax[0].set_title('Original Image')
ax[1].imshow(edge, cmap='gray')
ax[1].set_title('Edges detected by Canny Filter')
