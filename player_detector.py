import cv2
import numpy as np
from ultralytics import YOLO

class PlayerDetector:
    """
    玩家检测器 - 使用YOLO模型检测视频帧中的玩家位置
    支持在指定ROI区域内进行检测，并返回检测结果和中心点坐标
    """
    def __init__(self, model_path='weights/best.pt', conf_threshold=0.5):
        """
        初始化玩家检测器
        :param model_path: YOLO模型权重文件路径
        :param conf_threshold: 检测置信度阈值，低于此值的检测结果将被过滤
        """
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
    
    def detect(self, frame, roi=None):
        """
        在视频帧中检测玩家位置
        :param frame: 输入的视频帧（BGR格式）
        :param roi: 可选的感兴趣区域 (x1,y1,x2,y2)，如果提供则只在该区域内检测
        :return: 检测结果列表，每个元素包含bbox、center和confidence
        """
        # 如果指定了ROI，裁剪图像以加速检测
        if roi is not None:
            x1, y1, x2, y2 = roi
            frame = frame[y1:y2, x1:x2]
        
        # 使用YOLO模型进行目标检测
        results = self.model(frame, conf=self.conf_threshold, verbose=False)
        detections = []
        
        # 解析检测结果
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # 获取边界框坐标和置信度
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = box.conf[0].cpu().numpy()
                
                # 如果使用了ROI，需要将坐标转换回原始图像的坐标系
                if roi is not None:
                    x1 += roi[0]
                    y1 += roi[1]
                    x2 += roi[0]
                    y2 += roi[1]
                
                # 计算中心点坐标
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