import cv2
import numpy as np
import json
from block_rotation import rotate_block

def load_secret_key(secret_key_path):
    with open(secret_key_path, "r") as f:
        data = json.load(f)
    return data["metadata"], data["qr_shape"], data["total_bits"]

def extract_bits(stego_image, metadata, total_bits):
    bits = ""
    extracted_count = 0
    for block_info in metadata:
        x, y, angle = block_info
        block = stego_image[y:y+8, x:x+8]
        rotated_block = rotate_block(block, angle)

        for i in range(8):
            for j in range(8):
                if extracted_count < total_bits:
                    pixel = rotated_block[i, j]
                    bits += str(pixel[2] & 1)  # Blue channel LSB
                    extracted_count += 1
                else:
                    break
            if extracted_count >= total_bits:
                break

    if extracted_count != total_bits:
        print(f"⚠️ Warning: Extracted {extracted_count} bits, expected {total_bits}.")
    return bits

def reconstruct_qr(bits):
    # We dynamically calculate the size from number of bits
    length = int(np.sqrt(len(bits)))
    total_bits = length * length

    print(f"✅ Reconstructing QR with size {length}x{length}")

    bits = bits[:total_bits].ljust(total_bits, '0')
    arr = np.array([255 if b == '0' else 0 for b in bits], dtype=np.uint8)
    arr = arr.reshape((length, length))

    arr = cv2.resize(arr, (300, 300), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite("reconstructed_qr.png", arr)
    return arr

def decode_qr(qr_image_path):
    img = cv2.imread(qr_image_path)
    qr_decoder = cv2.QRCodeDetector()
    data, points, _ = qr_decoder.detectAndDecode(img)
    return data

if __name__ == "__main__":
    stego_image = cv2.imread("stego_output.png")
    metadata, qr_shape, total_bits = load_secret_key("secret_key.json")

    bits = extract_bits(stego_image, metadata, total_bits)

    qr_img = reconstruct_qr(bits)

    message = decode_qr("reconstructed_qr.png")
    print("✅ Decoded Message:", message)
