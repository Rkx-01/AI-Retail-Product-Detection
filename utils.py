import cv2
import numpy as np
import os

def draw_and_save_visuals(image: np.ndarray, products: list, output_path: str = "output.jpg"):
    """
    Draws colored bounding boxes based on group_id and saves the output.
    """
    # Create a copy to draw on
    vis_img = image.copy()
    
    # Generate random colors for up to 100 groups
    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(100, 3), dtype=int).tolist()
    
    for product in products:
        x1, y1, x2, y2 = product['bbox']
        group_id = product.get('group_id', -1)
        
        # Filter out boxes that are too large (likely whole shelf detections)
        if (x2-x1) * (y2-y1) > (vis_img.shape[0] * vis_img.shape[1] * 0.3):
            continue

        color = (128, 128, 128)
        if group_id >= 0:
            color = tuple(map(int, colors[group_id % len(colors)]))
            
        # Draw solid bounding box (thickness 2)
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 2)
        
        # Add a neat label at the top
        label = f"G-{group_id}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.35
        font_thickness = 1
        
        # Get text size for background box
        (tw, th), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
        
        # Draw filled background for the label
        cv2.rectangle(vis_img, (x1, y1 - th - 4), (x1 + tw + 4, y1), color, -1)
        
        # Draw label text
        # Decide text color based on background brightness
        brightness = (color[0] * 299 + color[1] * 587 + color[2] * 114) / 1000
        t_color = (0, 0, 0) if brightness > 127 else (255, 255, 255)
        
        cv2.putText(vis_img, label, (x1 + 2, y1 - 4), font, font_scale, t_color, font_thickness)
                    
    cv2.imwrite(output_path, vis_img)
    return output_path
