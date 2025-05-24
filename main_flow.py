import cv2
import json
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0 = all logs, 3 = errors only
from embed_qr import generate_qr_image, embed_qr_in_safe_blocks
from cnn_classifier import build_patch_classifier
from cnn_region_map import get_embedding_map, save_embedding_heatmap
from block_selector import get_safe_blocks

# Load cover image
img = cv2.imread("coverimage2.jpg")
model = build_patch_classifier()

# Optional: load weights if you trained it
# model.load_weights("patch_cnn_weights.h5")
def get_embedding_map(image, model):
    h, w = image.shape[:2]
    return np.random.choice([0, 1, 2], size=(h//8, w//8), p=[0.1, 0.4, 0.5])

    #for i in range(0, h - 8 + 1, 8):
        #if i % 80 == 0:
            #print(f"🔍 Processing row {i}/{h}")

# Step 1: Safe zone detection
heatmap = get_embedding_map(img, model)
save_embedding_heatmap(heatmap)

safe_mask = get_safe_blocks(img, model)
cv2.imwrite("safe_zone_map.png", safe_mask)

# Safe mask ready
safe_block_count = np.sum(safe_mask == 255) // (8 * 8)
max_bits_capacity = safe_block_count * 64
print(f"✅ Available bits capacity: {max_bits_capacity}")

# Generate QR
text = "Hi"  # <<< Use short text here!
qr_img = generate_qr_image(text)
print("✅ Generated QR shape:", qr_img.shape)

qr_bits_needed = len(qr_img.flatten())

if qr_bits_needed > max_bits_capacity:
    print("❌ QR too big for safe zones! Reduce message.")
    exit()

# Proceed to embed because QR fits
stego, metadata = embed_qr_in_safe_blocks(img, qr_img, safe_mask)
cv2.imwrite("stego_output.png", stego)


# Save secret key
metadata_serializable = {
    "metadata": [[int(x), int(y), int(angle)] for x, y, angle in metadata],
    "qr_shape": [qr_img.shape[0], qr_img.shape[1]],
    "total_bits": len(qr_img.flatten())
}
with open("secret_key.json", "w") as f:
    json.dump(metadata_serializable, f, indent=2)

print("✅ Stego image and secret key saved successfully.")