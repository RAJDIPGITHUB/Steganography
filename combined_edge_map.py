import cv2
import numpy as np

# Load edge-based map
edge_mask = cv2.imread("safe_zone_map.png", cv2.IMREAD_GRAYSCALE)

# Load CNN-based map
cnn_mask = cv2.imread("cnn_map_colored.png", cv2.IMREAD_GRAYSCALE)

# Resize CNN mask to match edge mask if needed
cnn_mask = cv2.resize(cnn_mask, (edge_mask.shape[1], edge_mask.shape[0]), interpolation=cv2.INTER_NEAREST)

# Threshold both masks to binary (0 or 255)
_, edge_bin = cv2.threshold(edge_mask, 127, 255, cv2.THRESH_BINARY)
_, cnn_bin = cv2.threshold(cnn_mask, 127, 255, cv2.THRESH_BINARY)

# Combine both maps using OR operation
combined_mask = cv2.bitwise_or(edge_bin, cnn_bin)

# Save the combined safe zone map
cv2.imwrite("combined_safe_zone_mask.png", combined_mask)
print("✅ Saved binary combined safe zone mask as 'combined_safe_zone_mask.png'")
