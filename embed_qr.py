import cv2
import numpy as np
from block_rotation import rotate_block
import qrcode

def generate_qr_image(text):
    qr = qrcode.QRCode(version=2, error_correction=qrcode.constants.ERROR_CORRECT_M)
    qr.add_data(text)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    return np.array(img.convert('L'))

def embed_qr_in_safe_blocks(image, qr_img, safe_mask):
    stego = image.copy()
    h, w, _ = image.shape
    idx = 0
    bits = ''.join(['1' if px < 128 else '0' for row in qr_img for px in row])
    metadata = []

    for y in range(0, h - 8 + 1, 8):
        for x in range(0, w - 8 + 1, 8):
            if idx >= len(bits):
                break

            if safe_mask[y, x] != 255:
                continue

            block = stego[y:y+8, x:x+8]
            angle = np.random.choice([0, 90, 180, 270])
            rotated = rotate_block(block.copy(), angle)

            for i in range(8):
                for j in range(8):
                    if idx >= len(bits):
                        break
                    pixel = list(rotated[i, j])
                    pixel[2] = np.uint8((int(pixel[2]) & ~1) | int(bits[idx]))
                    rotated[i, j] = pixel
                    idx += 1

            rotated = rotate_block(rotated, -angle % 360)
            stego[y:y+8, x:x+8] = rotated
            metadata.append([int(x), int(y), int(angle)])

    return stego, metadata