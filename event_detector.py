import cv2
import numpy as np
from abc import ABC, abstractmethod

class EventDetector(ABC):
    """事件检测器基类，定义所有事件检测器的通用接口"""
    
    @abstractmethod
    def detect(self, frame, frame_idx, **kwargs):
        """
        检测事件
        :param frame: 当前帧
        :param frame_idx: 帧索引
        :param kwargs: 其他参数
        :return: 检测到的事件列表 [{'event_type': str, 'details': dict, 'timestamp': float}]
        """
        pass


class FireEventDetector(EventDetector):
    """开火事件检测器（预留接口）- 用于检测玩家射击行为"""
    
    def __init__(self, conf_threshold=0.5):
        """
        :param conf_threshold: 检测置信度阈值
        """
        self.conf_threshold = conf_threshold
        # TODO: 加载开火检测模型
        # self.model = YOLO('weights/fire_detector.pt')
        pass
    
    def detect(self, frame, frame_idx, **kwargs):
        """
        检测开火事件
        TODO: 实现基于视觉特征的开火检测
        - 可以使用枪口火焰检测
        - 或者检测射击动作
        - 或者检测UI变化（如弹药减少）
        """
        events = []
        # TODO: 实现检测逻辑
        # detections = self.model(frame, conf=self.conf_threshold)
        # for det in detections:
        #     events.append({
        #         'frame_idx': frame_idx,
        #         'event_type': 'fire',
        #         'details': {'bbox': det['bbox'], 'confidence': det['conf']},
        #         'timestamp': frame_idx / kwargs.get('fps', 30)
        #     })
        return events


class PickupEventDetector(EventDetector):
    """拾取事件检测器（预留接口）- 用于检测玩家拾取物品行为"""
    
    def __init__(self, conf_threshold=0.5):
        self.conf_threshold = conf_threshold
        # TODO: 加载拾取检测模型
        # self.model = YOLO('weights/pickup_detector.pt')
        pass
    
    def detect(self, frame, frame_idx, **kwargs):
        """
        检测拾取物品事件
        TODO: 实现基于视觉特征的拾取检测
        - 检测物品拾取动画
        - 检测背包UI变化
        - 检测地面上的物品消失
        """
        events = []
        # TODO: 实现检测逻辑
        return events


class KillEventDetector(EventDetector):
    """击杀事件检测器（预留接口）- 用于检测玩家击杀敌人行为"""
    
    def __init__(self, conf_threshold=0.5):
        self.conf_threshold = conf_threshold
        # TODO: 加载击杀检测模型
        # self.model = YOLO('weights/kill_detector.pt')
        pass
    
    def detect(self, frame, frame_idx, **kwargs):
        """
        检测击杀事件
        TODO: 实现基于视觉特征的击杀检测
        - 检测击杀提示UI
        - 检测击杀回放画面
        - 检测击杀音效可视化
        """
        events = []
        # TODO: 实现检测逻辑
        return events


class DeathEventDetector(EventDetector):
    """死亡事件检测器（预留接口）- 用于检测玩家死亡行为"""
    
    def __init__(self, conf_threshold=0.5):
        self.conf_threshold = conf_threshold
        # TODO: 加载死亡检测模型
        # self.model = YOLO('weights/death_detector.pt')
        pass
    
    def detect(self, frame, frame_idx, **kwargs):
        """
        检测死亡事件
        TODO: 实现基于视觉特征的死亡检测
        - 检测死亡画面（如屏幕变灰）
        - 检测死亡UI提示
        - 检测重生画面
        """
        events = []
        # TODO: 实现检测逻辑
        return events


class EventDetectorManager:
    """事件检测管理器 - 统一管理多种事件检测器并协调检测流程"""
    
    def __init__(self, fps=30):
        self.fps = fps
        self.detectors = {}
        self.all_events = []
    
    def add_detector(self, event_type, detector):
        """
        添加事件检测器
        :param event_type: 事件类型名称
        :param detector: 事件检测器实例
        """
        self.detectors[event_type] = detector
        print(f"Added detector for event type: {event_type}")
    
    def detect_events(self, frame, frame_idx, roi=None, **kwargs):
        """
        对所有注册的事件检测器执行检测
        :param frame: 当前帧
        :param frame_idx: 帧索引
        :param roi: 可选的ROI区域，用于裁剪检测区域
        :param kwargs: 其他参数
        :return: 检测到的所有事件
        """
        # 如果指定了ROI，裁剪帧
        if roi is not None:
            x1, y1, x2, y2 = roi
            frame_roi = frame[y1:y2, x1:x2]
        else:
            frame_roi = frame
        
        # 对所有检测器执行检测
        detected_events = []
        for event_type, detector in self.detectors.items():
            events = detector.detect(frame_roi, frame_idx, fps=self.fps, **kwargs)
            # 如果使用了ROI，调整坐标
            if roi is not None:
                for event in events:
                    if 'details' in event and 'bbox' in event['details']:
                        x1, y1, x2, y2 = roi
                        bbox = event['details']['bbox']
                        event['details']['bbox'] = [
                            bbox[0] + x1,
                            bbox[1] + y1,
                            bbox[2] + x1,
                            bbox[3] + y1
                        ]
            detected_events.extend(events)
        
        self.all_events.extend(detected_events)
        return detected_events
    
    def get_all_events(self):
        """获取所有检测到的事件"""
        return self.all_events
    
    def save_events_csv(self, output_path):
        """保存事件到CSV文件"""
        import pandas as pd
        df = pd.DataFrame(self.all_events)
        if not df.empty:
            df.to_csv(output_path, index=False)
        return df


def create_default_event_detectors():
    """
    创建默认的事件检测器实例
    返回一个EventDetectorManager实例，包含所有可用的事件检测器
    目前所有检测器都是预留接口，等待后续实现具体检测逻辑
    """
    manager = EventDetectorManager(fps=30)
    
    # 添加各种事件检测器
    manager.add_detector('fire', FireEventDetector())
    manager.add_detector('pickup', PickupEventDetector())
    manager.add_detector('kill', KillEventDetector())
    manager.add_detector('death', DeathEventDetector())
    
    return manager
