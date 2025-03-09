import cv2
from ultralytics import YOLO

MODEL_PATH = 'artifacts/models/best.pt'

def detect_bubbles_with_yolo(image_path):
    # Load pre-trained YOLO model
    model = YOLO(MODEL_PATH)  # Path to your trained model
    
    # Perform detection
    results = model(image_path)
    
    # Load original image
    image = cv2.imread(image_path)
    result = image.copy()
    
    # Process detections
    text_regions = []
    
    # Process the first result (assuming single image input)
    for i, detection in enumerate(results[0].boxes.data):
        x1, y1, x2, y2, confidence, class_id = detection.tolist()
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        # Draw rectangle
        cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Store region info
        text_regions.append({
            "id": i,
            "type": "Speech bubble",
            "bbox": (x1, y1, x2-x1, y2-y1),
            "confidence": confidence
        })
        
        # Label
        cv2.putText(result, f"Bubble #{i} ({confidence:.2f})", (x1, y1-5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return result, text_regions
