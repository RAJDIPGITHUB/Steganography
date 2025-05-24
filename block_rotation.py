import cv2

def rotate_block(block, angle):
    if angle == 90:
        return cv2.rotate(block, cv2.ROTATE_90_CLOCKWISE)
    elif angle == 180:
        return cv2.rotate(block, cv2.ROTATE_180)
    elif angle == 270:
        return cv2.rotate(block, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return block