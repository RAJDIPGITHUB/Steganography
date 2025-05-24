import cv2
import numpy as np

def calculate_psnr(img1, img2):
    mse = np.mean((img1.astype(np.float32) - img2.astype(np.float32)) ** 2)
    if mse == 0:
        return float('inf')  # No difference
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

cover = cv2.imread("coverimage1.jpg")
stego = cv2.imread("stego_output.png")
psnr_value = calculate_psnr(cover, stego)
print("✅ PSNR:", psnr_value, "dB")
