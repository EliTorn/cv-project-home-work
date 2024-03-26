import cv2
import numpy as np


def stitch_panorama(images):
    stitcher = cv2.Stitcher_create(mode=cv2.Stitcher_SCANS)
    image_arrays = [cv2.imread(img) for img in images] if isinstance(images[0], str) else images

    keypoints, descriptors = [], []
    for image in image_arrays:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        sift = cv2.SIFT_create()  # Use cv2.SIFT_create() for OpenCV 4.4.0+
        kp, des = sift.detectAndCompute(gray, None)
        keypoints.append(kp)
        descriptors.append(des)

    matches = []
    for i in range(len(images) - 1):
        flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
        matches12 = flann.knnMatch(descriptors[i], descriptors[i + 1], k=2)  # Access first list of matches
        good_matches = []
        for m, n in matches12:  # Iterate over matches for the first pair
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)  # Append individual DMatch object

        matches.append(good_matches)  # Append list of good matches

    if len(good_matches) < 4:
        print("Not enough matches found - Panorama stitching failed!")
        return None

    # Calculate homography based on good matches from the first image pair
    src_pts = np.float32([keypoints[0][m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints[1][m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # Stitch the images together
    status, panorama = stitcher.stitch(image_arrays, [homography])  # Provide homography in a list

    if status != cv2.Stitcher_OK:
        print("Panorama stitching failed with status:", status)
        return None

    return panorama


# Example usage (replace with your image paths)
image_paths = ["image_text.png", "image_text.png", "image_text.png"]
panorama = stitch_panorama(image_paths)

if panorama is not None:
    cv2.imshow("Panorama", panorama)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
