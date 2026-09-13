"""
Program 15: Frequency-Domain High-Pass Filtering
- Creates and applies a high-pass frequency mask (suppresses central low
  frequencies, preserves higher-frequency content)
- Performs the required inverse operations
- Saves the reconstructed HPF output as output.png
"""

import cv2
import numpy as np

# Read grayscale image
gray = cv2.imread("input.jpg", cv2.IMREAD_GRAYSCALE)
rows, cols = gray.shape
crow, ccol = rows // 2, cols // 2  # center coordinates of the spectrum

# Compute DFT and shift zero-frequency to center
float_img = np.float32(gray)
dft = cv2.dft(float_img, flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shifted = np.fft.fftshift(dft)

# Create a High-Pass mask: the inverse of the low-pass mask. A circular
# region of 0s at the center (blocks low frequencies) and 1s elsewhere
# (keeps high frequencies, which correspond to edges and fine detail).
RADIUS = 40  # frequencies within this radius of the center are suppressed
mask = np.ones((rows, cols, 2), np.uint8)
y, x = np.ogrid[:rows, :cols]
mask_area = (x - ccol) ** 2 + (y - crow) ** 2 <= RADIUS ** 2
mask[mask_area] = 0

# Apply the mask to the shifted DFT (element-wise, keeps only high freqs)
filtered_dft = dft_shifted * mask

# Inverse operations: shift back, then compute inverse DFT
dft_ishifted = np.fft.ifftshift(filtered_dft)
img_back = cv2.idft(dft_ishifted)
img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

# Normalize the reconstructed image back to a displayable 0-255 range
img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX)
img_back = img_back.astype(np.uint8)

# Save the high-pass filtered (edge-emphasized) reconstruction
cv2.imwrite("output.png", img_back)
print(f"Applied high-pass frequency mask with radius {RADIUS}")
print("Saved frequency-domain HPF output as output.png")
