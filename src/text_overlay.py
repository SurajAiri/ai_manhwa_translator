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


def wrap_text_in_rectangle(image, text, bbox, font_face=cv2.FONT_HERSHEY_SIMPLEX, 
                          base_font_scale=1.0, color=(0, 0, 0), thickness=3, 
                          bg_color=(255, 255, 255), padding=10):
    """
    Wraps text to fit within a rectangle on an image with flexible font sizing.
    
    Args:
        image: The input image (numpy array)
        text: The text to be overlaid
        bbox: Tuple or list containing (x, y, w, h) coordinates of the rectangle
        font_face: OpenCV font face
        base_font_scale: Initial font scale to try
        color: Text color as BGR tuple
        thickness: Text thickness
        bg_color: Background color as BGR tuple
        padding: Padding inside the rectangle
        
    Returns:
        Modified image with text wrapped in rectangle
    """
    result = image.copy()
    x, y, w, h = bbox
    
    # Apply padding to the effective area
    x_pad = x + padding
    y_pad = y + padding
    w_pad = w - (2 * padding)
    h_pad = h - (2 * padding)
    
    # Draw background rectangle
    cv2.rectangle(result, (x, y), (x + w, y + h), bg_color, -1)
    
    if not text or w_pad <= 0 or h_pad <= 0:
        return result
    
    # Split text into words
    words = text.split()
    if not words:
        return result
    
    # Try different font scales to find the optimal one
    max_font_scale = 2.0  # Maximum font scale to try
    min_font_scale = 0.4  # Minimum font scale
    optimal_font_scale = min_font_scale
    optimal_lines = []
    
    # Start with a large font and decrease until text fits
    font_scale = max_font_scale
    while font_scale >= min_font_scale:
        # Try to fit text with current font scale
        current_lines = []
        current_line = words[0]
        
        for word in words[1:]:
            # Check if adding this word exceeds the width
            test_line = current_line + " " + word
            text_size = cv2.getTextSize(test_line, font_face, font_scale, thickness)
            
            if text_size[0][0] <= w_pad:
                current_line = test_line  # Add word to current line
            else:
                current_lines.append(current_line)  # Store the current line
                current_line = word  # Start a new line with the current word
        
        # Add the last line
        current_lines.append(current_line)
        
        # Calculate total height needed
        total_height = 0
        for line in current_lines:
            text_size = cv2.getTextSize(line, font_face, font_scale, thickness)
            line_height = text_size[0][1]
            total_height += line_height + padding  # Add spacing between lines
        
        if total_height > 0:
            total_height -= padding  # Remove extra padding from last line
            
        # If text fits with current font scale
        if total_height <= h_pad:
            optimal_font_scale = font_scale
            optimal_lines = current_lines
            break
            
        # Reduce font scale and try again
        font_scale -= 0.1
    
    # If we couldn't fit the text even with minimum font size, use the minimum and truncate
    if not optimal_lines:
        optimal_font_scale = min_font_scale
        # Recalculate lines with minimum font scale
        optimal_lines = []
        current_line = words[0]
        
        for word in words[1:]:
            test_line = current_line + " " + word
            text_size = cv2.getTextSize(test_line, font_face, min_font_scale, thickness)
            
            if text_size[0][0] <= w_pad:
                current_line = test_line
            else:
                optimal_lines.append(current_line)
                current_line = word
                
        optimal_lines.append(current_line)
    
    # Calculate line heights and total height for proper vertical distribution
    line_heights = []
    total_height = 0
    for line in optimal_lines:
        text_size = cv2.getTextSize(line, font_face, optimal_font_scale, thickness)
        line_height = text_size[0][1]
        line_heights.append(line_height)
        total_height += line_height
    
    # Add spacing between lines
    total_height += (len(optimal_lines) - 1) * padding
    
    # Calculate starting Y position to center text block vertically
    y_start = y_pad + (h_pad - total_height) // 2
    
    # Now draw the text
    y_pos = y_start
    for i, line in enumerate(optimal_lines):
        text_size = cv2.getTextSize(line, font_face, optimal_font_scale, thickness)
        text_width = text_size[0][0]
        
        # Center text horizontally
        text_x = x_pad + (w_pad - text_width) // 2
        
        # Position text vertically
        text_y = y_pos + line_heights[i]
        
        cv2.putText(result, line, (text_x, text_y), font_face, optimal_font_scale, color, thickness)
        
        # Update y_pos for next line
        y_pos += line_heights[i] + padding
    
    return result

def draw_rounded_rectangle(image, top_left, bottom_right, color, radius=10, thickness=-1):
    """
    Draw a rounded rectangle on an image.
    
    Args:
        image: The input image
        top_left: (x, y) coordinates of the top-left point
        bottom_right: (x, y) coordinates of the bottom-right point
        color: Rectangle color as BGR tuple
        radius: Corner radius
        thickness: Thickness of the rectangle border. Negative means filled rectangle
        
    Returns:
        Image with the rounded rectangle
    """
    # Draw the main rectangle
    x1, y1 = top_left
    x2, y2 = bottom_right
    
    # Make sure radius is not too large
    radius = min(radius, (x2 - x1) // 2, (y2 - y1) // 2)
    
    # Draw the main filled rectangle with the corners cut out
    cv2.rectangle(image, (x1 + radius, y1), (x2 - radius, y2), color, thickness)
    cv2.rectangle(image, (x1, y1 + radius), (x2, y2 - radius), color, thickness)
    
    # Draw the four corner circles
    if thickness < 0:  # Filled rectangle
        cv2.circle(image, (x1 + radius, y1 + radius), radius, color, -1)
        cv2.circle(image, (x1 + radius, y2 - radius), radius, color, -1)
        cv2.circle(image, (x2 - radius, y1 + radius), radius, color, -1)
        cv2.circle(image, (x2 - radius, y2 - radius), radius, color, -1)
    else:  # Outlined rectangle
        cv2.circle(image, (x1 + radius, y1 + radius), radius, color, thickness)
        cv2.circle(image, (x1 + radius, y2 - radius), radius, color, thickness)
        cv2.circle(image, (x2 - radius, y1 + radius), radius, color, thickness)
        cv2.circle(image, (x2 - radius, y2 - radius), radius, color, thickness)
        
    return image

def wrap_text_in_rounded_rectangle(image, text, bbox, font_face=cv2.FONT_HERSHEY_SIMPLEX, 
                          base_font_scale=1.0, color=(0, 0, 0), thickness=3, 
                          bg_color=(255, 255, 255), padding=10, corner_radius=150):
    """
    Wraps text to fit within a rounded rectangle on an image with flexible font sizing.
    
    Args:
        image: The input image (numpy array)
        text: The text to be overlaid
        bbox: Tuple or list containing (x, y, w, h) coordinates of the rectangle
        font_face: OpenCV font face
        base_font_scale: Initial font scale to try
        color: Text color as BGR tuple
        thickness: Text thickness
        bg_color: Background color as BGR tuple
        padding: Padding inside the rectangle
        corner_radius: Radius for rounded corners
        
    Returns:
        Modified image with text wrapped in rounded rectangle
    """
    result = image.copy()
    x, y, w, h = bbox
    
    # Apply padding to the effective area
    x_pad = x + padding
    y_pad = y + padding
    w_pad = w - (2 * padding)
    h_pad = h - (2 * padding)
    
    # Draw rounded background rectangle
    draw_rounded_rectangle(result, (x, y), (x + w, y + h), (0,0,0), radius=corner_radius, thickness=-1)
    # Draw outer black rounded rectangle
    draw_rounded_rectangle(result, (x, y), (x + w, y + h), (0,0,0), radius=corner_radius, thickness=-1)
    
    # Draw inner white rounded rectangle with 4px smaller dimensions
    border_gap = 4
    draw_rounded_rectangle(result, 
                          (x + border_gap, y + border_gap), 
                          (x + w - border_gap, y + h - border_gap), 
                          bg_color, 
                          radius=corner_radius - border_gap, 
                          thickness=-1)

    
    if not text or w_pad <= 0 or h_pad <= 0:
        return result
    
    # Split text into words
    words = text.split()
    if not words:
        return result
    
    # Try different font scales to find the optimal one
    max_font_scale = 2.0  # Maximum font scale to try
    min_font_scale = 0.4  # Minimum font scale
    optimal_font_scale = min_font_scale
    optimal_lines = []
    
    # Start with a large font and decrease until text fits
    font_scale = max_font_scale
    while font_scale >= min_font_scale:
        # Try to fit text with current font scale
        current_lines = []
        current_line = words[0]
        
        for word in words[1:]:
            # Check if adding this word exceeds the width
            test_line = current_line + " " + word
            text_size = cv2.getTextSize(test_line, font_face, font_scale, thickness)
            
            if text_size[0][0] <= w_pad:
                current_line = test_line  # Add word to current line
            else:
                current_lines.append(current_line)  # Store the current line
                current_line = word  # Start a new line with the current word
        
        # Add the last line
        current_lines.append(current_line)
        
        # Calculate total height needed
        total_height = 0
        for line in current_lines:
            text_size = cv2.getTextSize(line, font_face, font_scale, thickness)
            line_height = text_size[0][1]
            total_height += line_height + padding  # Add spacing between lines
        
        if total_height > 0:
            total_height -= padding  # Remove extra padding from last line
            
        # If text fits with current font scale
        if total_height <= h_pad:
            optimal_font_scale = font_scale
            optimal_lines = current_lines
            break
            
        # Reduce font scale and try again
        font_scale -= 0.1
    
    # If we couldn't fit the text even with minimum font size, use the minimum and truncate
    if not optimal_lines:
        optimal_font_scale = min_font_scale
        # Recalculate lines with minimum font scale
        optimal_lines = []
        current_line = words[0]
        
        for word in words[1:]:
            test_line = current_line + " " + word
            text_size = cv2.getTextSize(test_line, font_face, min_font_scale, thickness)
            
            if text_size[0][0] <= w_pad:
                current_line = test_line
            else:
                optimal_lines.append(current_line)
                current_line = word
                
        optimal_lines.append(current_line)
    
    # Calculate line heights and total height for proper vertical distribution
    line_heights = []
    total_height = 0
    for line in optimal_lines:
        text_size = cv2.getTextSize(line, font_face, optimal_font_scale, thickness)
        line_height = text_size[0][1]
        line_heights.append(line_height)
        total_height += line_height
    
    # Add spacing between lines
    total_height += (len(optimal_lines) - 1) * padding
    
    # Calculate starting Y position to center text block vertically
    y_start = y_pad + (h_pad - total_height) // 2
    
    # Now draw the text
    y_pos = y_start
    for i, line in enumerate(optimal_lines):
        text_size = cv2.getTextSize(line, font_face, optimal_font_scale, thickness)
        text_width = text_size[0][0]
        
        # Center text horizontally
        text_x = x_pad + (w_pad - text_width) // 2
        
        # Position text vertically
        text_y = y_pos + line_heights[i]
        
        cv2.putText(result, line, (text_x, text_y), font_face, optimal_font_scale, color, thickness)
        
        # Update y_pos for next line
        y_pos += line_heights[i] + padding
    
    return result

def overlay_text(image, extracted_texts):
    """
    Overlays text on the image with white background and flexible font size
    
    Args:
        image: The input image
        extracted_texts: List of dictionaries containing bbox information and translated text
    
    Returns:
        Image with overlaid text
    """
    result = image.copy()
    
    for i, ext in enumerate(extracted_texts):
        bbox = ext['bbox']
        text = ext['translated_text']
        result = wrap_text_in_rounded_rectangle(result, text, bbox)
    
    return result