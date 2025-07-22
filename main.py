import cv2
from skimage.metrics import structural_similarity as ssim

# Step 1: Load Images (Grayscale for simplicity)
img1 = cv2.imread('1.png', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('4.png', cv2.IMREAD_GRAYSCALE)

# Step 2: Resize Images to the same size
img1 = cv2.resize(img1, (256, 256))
img2 = cv2.resize(img2, (256, 256))

# Step 3: Calculate SSIM Score
similarity_index, diff = ssim(img1, img2, full=True)

print(f"Similarity Score: {similarity_index:.2f}")
