import cv2
import numpy as np
from cnn_region_map import get_embedding_map

def get_edge_map(image, threshold=40):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Sobel(gray, cv2.CV_64F, 1, 1, ksize=3)
    edge_map = cv2.convertScaleAbs(edges)
    binary_map = np.zeros_like(edge_map)
    binary_map[edge_map > threshold] = 255
    return binary_map

def get_safe_blocks(image, cnn_model, combine='intersection'):
    edge_map = get_edge_map(image)
    cnn_map = get_embedding_map(image, cnn_model)

    h, w = edge_map.shape
    mask = np.zeros((h, w), dtype=np.uint8)

    for y in range(0, h - 8 + 1, 8):
        for x in range(0, w - 8 + 1, 8):
            edge_block = edge_map[y:y+8, x:x+8]
            cnn_class = cnn_map[y//8, x//8]

            edge_pass = np.std(edge_block) > 40
            cnn_pass = cnn_class == 2

            use = edge_pass and cnn_pass

            if use:
                mask[y:y+8, x:x+8] = 255

    return mask