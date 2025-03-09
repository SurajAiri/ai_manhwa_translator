import numpy as np
from matplotlib import pyplot as plt
import cv2
def text_padding(bbox, padRatio=0.1):
    """
    Adjusts the bounding box coordinates by adding padding
    
    Args:
        bbox: Tuple or list containing (x, y, w, h) coordinates
        padRatio: Ratio of padding to be added
        
    Returns:
        Adjusted coordinates (x, y, w, h) with padding
    """
    x1, y1, w, h = bbox

    # Calculate padding values
    padX = int(w * padRatio)
    padY = int(h * padRatio)

    # Apply padding
    x1 += padX
    y1 += padY
    w -= padX * 2
    h -= padY * 2

    return x1, y1, w, h
    


img = np.zeros((100, 200, 3), dtype=np.uint8)
bbox = (0, 0,  160,50)
cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), (0, 255, 0), 2)
plt.imshow(img)

bbox = text_padding(bbox, padRatio=0.1)
cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), (0, 255, 255), 2)
plt.imshow(img)
