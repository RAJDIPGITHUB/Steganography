import numpy as np
import cv2
import matplotlib.pyplot as plt


def get_embedding_map(image, model=None):
    h, w = image.shape[:2]
    return np.random.choice([0, 1, 2], size=(h//8, w//8), p=[0.2, 0.5, 0.3])


    for i in range(0, h - 8 + 1, 8):
        for j in range(0, w - 8 + 1, 8):
            patch = gray[i:i+8, j:j+8]
            patch = patch.reshape((1, 8, 8, 1)) / 255.0
            pred = np.argmax(model.predict(patch, verbose=0))
            heatmap[i//8, j//8] = pred

    return heatmap

def save_embedding_heatmap(heatmap, filename="cnn_map_colored.png"):
    plt.figure(figsize=(6, 6))
    plt.imshow(heatmap, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.title("CNN-Based Embedding Heatmap")
    plt.savefig(filename)
    plt.close()