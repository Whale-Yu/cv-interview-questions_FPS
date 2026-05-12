import cv2
import numpy as np
from ultralytics import YOLO

class PlayerDetector:
    def __init__(self, model_path='weights/best.pt', conf_threshold=0.5):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
    
    def detect(self, frame, roi=None):
        if roi is not None:
            x1, y1, x2, y2 = roi
            frame = frame[y1:y2, x1:x2]
        
        results = self.model(frame, conf=self.conf_threshold, verbose=False)
        detections = []
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = box.conf[0].cpu().numpy()
                
                if roi is not None:
                    x1 += roi[0]
                    y1 += roi[1]
                    x2 += roi[0]
                    y2 += roi[1]
                
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                
                detections.append({
                    'bbox': [x1, y1, x2, y2],
                    'center': (cx, cy),
                    'confidence': float(conf)
                })
        
        return detections

if __name__ == '__main__':
    detector = PlayerDetector()
    cap = cv2.VideoCapture("videos/20260510_163202.mp4")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        detections = detector.detect(frame)
        print(detections)
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        from utils.roi_extractor import resize_image_for_display
        display_frame, _ = resize_image_for_display(frame)
        
        cv2.imshow("Player Detector", display_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()