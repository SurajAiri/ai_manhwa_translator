import cv2
import numpy as np

def inpaint_text(image, text_regions):
    # Create a mask for inpainting
    mask = np.zeros(image.shape[:2], dtype=np.uint8)

    for region in text_regions:
        x, y, w, h = region['bbox']
        mask[y:y+h, x:x+w] = 255  # Mark the text area in the mask

    # Inpaint the image using the mask
    inpainted_image = cv2.inpaint(image, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    
    return inpainted_image

# Example usage
# inpainted_image = inpaint_text(result_image, text_regions)