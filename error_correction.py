import cv2
import numpy as np
from block_rotation import rotate_block

def robust_extract_qr(stego_img, metadata, qr_shape, copies=5):
    # Initialize voting system with confidence tracking
    votes = np.zeros((qr_shape[0], qr_shape[1], 2), dtype=np.float32)  # votes[0] and votes[1]
    block_size = 8
    bits_per_copy = (qr_shape[0] * qr_shape[1]) // (block_size * block_size)
    
    for copy_num in range(copies):
        bit_ptr = 0
        for x, y, angle in metadata:
            if bit_ptr >= bits_per_copy * block_size * block_size:
                break
                
            block = stego_img[y:y+block_size, x:x+block_size]
            if block.shape[:2] != (block_size, block_size):
                continue
                
            rotated = rotate_block(block, angle)
            
            # Calculate position in QR code
            px = (bit_ptr // block_size) % qr_shape[1]
            py = (bit_ptr // (block_size * qr_shape[1])) * block_size
            
            for i in range(block_size):
                for j in range(block_size):
                    if py+i >= qr_shape[0] or px+j >= qr_shape[1]:
                        continue
                        
                    # Extract LSB from all channels with confidence
                    blue = rotated[i, j, 2] & 1
                    green = rotated[i, j, 1] & 1
                    red = rotated[i, j, 0] & 1
                    
                    # Weighted voting (blue channel most important)
                    votes[py+i, px+j, blue] += 0.6  # Blue channel weight
                    votes[py+i, px+j, green] += 0.2  # Green channel weight
                    votes[py+i, px+j, red] += 0.2  # Red channel weight
                    
                    bit_ptr += 1
    
    # Reconstruct QR with confidence-based voting
    reconstructed = np.zeros(qr_shape, dtype=np.uint8)
    for y in range(qr_shape[0]):
        for x in range(qr_shape[1]):
            total_votes = votes[y, x, 0] + votes[y, x, 1]
            if total_votes == 0:
                # No info - use default white (1)
                reconstructed[y, x] = 255
            else:
                # Confidence-based decision
                confidence_0 = votes[y, x, 0] / total_votes
                confidence_1 = votes[y, x, 1] / total_votes
                
                if abs(confidence_0 - confidence_1) < 0.2:  # Too close to call
                    # Use neighborhood analysis
                    neighborhood = votes[max(0,y-1):min(qr_shape[0],y+2),
                                    max(0,x-1):min(qr_shape[1],x+2)]
                    if np.sum(neighborhood[:,:,0]) > np.sum(neighborhood[:,:,1]):
                        reconstructed[y, x] = 0
                    else:
                        reconstructed[y, x] = 255
                else:
                    reconstructed[y, x] = 0 if confidence_0 > confidence_1 else 255
    
    # Enhanced post-processing
    reconstructed = cv2.medianBlur(reconstructed, 3)
    
    # QR code alignment markers enhancement
    marker_size = 7  # Standard QR alignment marker size
    for y in [0, qr_shape[0]-marker_size]:
        for x in [0, qr_shape[1]-marker_size]:
            reconstructed[y:y+marker_size, x:x+marker_size] = cv2.rectangle(
                reconstructed[y:y+marker_size, x:x+marker_size],
                (0,0), (marker_size-1,marker_size-1), 0, -1
            )
            reconstructed[y+2:y+marker_size-2, x+2:x+marker_size-2] = cv2.rectangle(
                reconstructed[y+2:y+marker_size-2, x+2:x+marker_size-2],
                (0,0), (marker_size-5,marker_size-5), 255, -1
            )
    
    return reconstructed