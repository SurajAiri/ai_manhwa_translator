import cv2
import os

def crop_text_regions(image, text_regions, output_dir='cropped_texts'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cropped_images = []
    
    for region in text_regions:
        x, y, w, h = region['bbox']
        cropped_image = image[y:y+h, x:x+w]
        
        # Save the cropped image
        cropped_image_path = os.path.join(output_dir, f"bubble_{region['id']}.png")
        cv2.imwrite(cropped_image_path, cropped_image)
        
        # Store the cropped image and its position
        cropped_images.append({
            "id": region['id'],
            "image": cropped_image,
            "path": cropped_image_path,
            "bbox": region['bbox']
        })
    
    return cropped_images
