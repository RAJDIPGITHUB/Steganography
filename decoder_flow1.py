import cv2
import numpy as np
import json
from block_rotation import rotate_block

def load_secret_key(secret_key_path):
    with open(secret_key_path, "r") as f:
        data = json.load(f)
    return data["metadata"], data["qr_shape"], data["total_bits"]

def extract_qr_from_blocks(stego_image, metadata, qr_shape):
    qr_accumulator = np.zeros(qr_shape, dtype=np.float32)
    count_map = np.zeros(qr_shape, dtype=np.uint8)

    for idx, (x, y, angle) in enumerate(metadata):
        block = stego_image[y:y+8, x:x+8]
        if block.shape != (8, 8, 3):
            continue

        rotated_block = rotate_block(block, angle)

        # Extract 1-bit LSB from blue channel
        bit_block = np.array([[pixel[2] & 1 for pixel in row] for row in rotated_block], dtype=np.uint8)
        bit_block = bit_block * 255  # Convert to binary image

        block_x = (idx * 8) % qr_shape[1]
        block_y = ((idx * 8) // qr_shape[1]) * 8

        if block_y + 8 <= qr_shape[0] and block_x + 8 <= qr_shape[1]:
            qr_accumulator[block_y:block_y+8, block_x:block_x+8] += bit_block
            count_map[block_y:block_y+8, block_x:block_x+8] += 1

    count_map[count_map == 0] = 1
    averaged_qr = (qr_accumulator / count_map).astype(np.uint8)

    return averaged_qr

def decode_qr(qr_img_arr):
    temp_path = "reconstructed_qr1.png"
    resized = cv2.resize(qr_img_arr, (300, 300), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(temp_path, resized)
    qr_decoder = cv2.QRCodeDetector()
    data, _, _ = qr_decoder.detectAndDecode(resized)
    return data

if __name__ == "__main__":
    stego_image = cv2.imread("compressed_stego_output.png")  # ← use compressed
    metadata, qr_shape, _ = load_secret_key("secret_key.json")

    qr_image = extract_qr_from_blocks(stego_image, metadata, qr_shape)
    decoded_msg = decode_qr(qr_image)
    print("✅ Decoded Message:", decoded_msg)
