import cv2
from skimage.metrics import structural_similarity as ssim

def calculate_ssim(img1, img2):
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    score, _ = ssim(img1_gray, img2_gray, full=True)
    return score

cover = cv2.imread("coverimage1.jpg")
stego = cv2.imread("stego_output.png")
ssim_score = calculate_ssim(cover, stego)
print("✅ SSIM:", ssim_score)