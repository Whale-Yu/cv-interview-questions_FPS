import numpy as np
from filterpy.kalman import KalmanFilter

class KalmanBoxTracker:
    """
    卡尔曼滤波框跟踪器 - 用于跟踪单个目标的边界框
    使用7维状态空间：[x, y, w, h, vx, vy, vz]，其中x,y是左上角坐标，w,h是宽高
    """
    def __init__(self, bbox):
        """
        初始化卡尔曼滤波器跟踪器
        :param bbox: 初始边界框 [x1, y1, x2, y2]
        """
        # 创建7维状态的卡尔曼滤波器（位置+速度）
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        
        # 状态转移矩阵 F：描述状态如何随时间变化
        self.kf.F = np.array([[1,0,0,0,1,0,0],
                              [0,1,0,0,0,1,0],
                              [0,0,1,0,0,0,1],
                              [0,0,0,1,0,0,0],
                              [0,0,0,0,1,0,0],
                              [0,0,0,0,0,1,0],
                              [0,0,0,0,0,0,1]])
        
        # 观测矩阵 H：将7维状态映射到4维观测值（x, y, w, h）
        self.kf.H = np.array([[1,0,0,0,0,0,0],
                              [0,1,0,0,0,0,0],
                              [0,0,1,0,0,0,0],
                              [0,0,0,1,0,0,0]])
        
        # 协方差矩阵初始化
        self.kf.P[4:,4:] *= 1000.0  # 速度的不确定性较大
        self.kf.P *= 10.0
        
        # 测量噪声协方差
        self.kf.R = np.array([[10, 0, 0, 0],
                             [0, 10, 0, 0],
                             [0, 0, 10, 0],
                             [0, 0, 0, 10]])
        
        # 过程噪声协方差
        self.kf.Q[-1,-1] *= 0.01
        self.kf.Q[4:,4:] *= 0.01
        
        # 初始化状态向量：[x, y, w, h, 0, 0, 0]
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        self.kf.x[:4] = np.array([[x1], [y1], [w], [h]])
        
        # 跟踪器属性
        self.id = None  # 跟踪ID
        self.age = 0  # 跟踪器存活帧数
        self.hits = 0  # 成功匹配次数
        self.time_since_update = 0  # 距离上次更新的帧数
    
    def update(self, bbox):
        """
        使用新的检测结果更新卡尔曼滤波器
        :param bbox: 新的边界框检测 [x1, y1, x2, y2]
        """
        self.time_since_update = 0  # 重置未更新计数器
        self.hits += 1  # 增加成功匹配次数
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        # 用观测值更新滤波器状态
        self.kf.update(np.array([[x1], [y1], [w], [h]]))
    
    def predict(self):
        """
        预测下一帧的目标位置
        :return: 预测的边界框 [x1, y1, x2, y2]
        """
        # 确保高度不为负数
        if self.kf.x[6] + self.kf.x[2] <= 0:
            self.kf.x[6] *= 0.0
        # 执行卡尔曼滤波预测步骤
        self.kf.predict()
        self.age += 1  # 增加年龄
        if self.time_since_update > 0:
            self.hits = 0  # 如果连续未更新，重置hits计数
        self.time_since_update += 1
        return self.get_state()
    
    def get_state(self):
        """
        获取当前估计的目标状态
        :return: 边界框坐标 [x1, y1, x2, y2]
        """
        x = self.kf.x[:4].flatten()
        return [int(x[0]), int(x[1]), int(x[0]+x[2]), int(x[1]+x[3])]

class TrackerManager:
    """
    跟踪管理器 - 管理多个卡尔曼滤波跟踪器，实现多目标跟踪
    使用IOU（交并比）进行检测与跟踪器的匹配
    """
    def __init__(self, max_age=30, min_hits=1):
        """
        初始化跟踪管理器
        :param max_age: 跟踪器最大存活帧数（未检测到目标时）
        :param min_hits: 跟踪器变为活跃状态所需的最小匹配次数
        """
        self.trackers = []  # 所有跟踪器列表
        self.max_age = max_age
        self.min_hits = min_hits
        self.next_id = 1  # 下一个分配的跟踪ID
    
    def iou(self, boxA, boxB):
        """
        计算两个边界框的交并比（IOU）
        :param boxA: 第一个边界框 [x1, y1, x2, y2]
        :param boxB: 第二个边界框 [x1, y1, x2, y2]
        :return: IOU值（0-1之间）
        """
        x1, y1, x2, y2 = boxA
        x3, y3, x4, y4 = boxB
        
        # 计算交集区域
        inter_x1 = max(x1, x3)
        inter_y1 = max(y1, y3)
        inter_x2 = min(x2, x4)
        inter_y2 = min(y2, y4)
        
        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
        areaA = (x2 - x1) * (y2 - y1)
        areaB = (x4 - x3) * (y4 - y3)
        
        # 计算IOU
        return inter_area / float(areaA + areaB - inter_area) if (areaA + areaB - inter_area) > 0 else 0
    
    def update(self, detections):
        """
        更新所有跟踪器并处理新的检测结果
        :param detections: 当前帧的检测结果列表
        :return: 活跃跟踪列表，每个元素包含id、bbox和center
        """
        # 第一步：对所有现有跟踪器进行预测
        for tracker in self.trackers:
            tracker.predict()
        
        # 第二步：使用贪心算法匹配检测与跟踪器
        matched = []  # 已匹配的(跟踪器索引, 检测索引)对
        unmatched_detections = list(range(len(detections)))  # 未匹配的检测
        unmatched_trackers = list(range(len(self.trackers)))  # 未匹配的跟踪器
        
        # 为每个跟踪器寻找最佳匹配的检测
        for t_idx, tracker in enumerate(self.trackers):
            best_iou = 0
            best_d_idx = -1
            
            for d_idx in unmatched_detections:
                iou = self.iou(tracker.get_state(), detections[d_idx]['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_d_idx = d_idx
            
            # 如果IOU超过阈值，则匹配成功
            if best_iou > 0.3 and best_d_idx != -1:
                tracker.update(detections[best_d_idx]['bbox'])
                matched.append((t_idx, best_d_idx))
                unmatched_detections.remove(best_d_idx)
                unmatched_trackers.remove(t_idx)
        
        # 第三步：为未匹配的检测创建新的跟踪器
        for d_idx in unmatched_detections:
            new_tracker = KalmanBoxTracker(detections[d_idx]['bbox'])
            new_tracker.id = self.next_id
            self.next_id += 1
            self.trackers.append(new_tracker)
        
        # 第四步：移除长时间未更新的跟踪器
        self.trackers = [t for t in self.trackers if t.time_since_update <= self.max_age]
        
        # 第五步：收集活跃的跟踪结果
        active_tracks = []
        for tracker in self.trackers:
            # 跟踪器需要满足最小命中次数或最近刚更新过
            if tracker.hits >= self.min_hits or tracker.time_since_update <= 1:
                state = tracker.get_state()
                cx = (state[0] + state[2]) // 2
                cy = (state[1] + state[3]) // 2
                active_tracks.append({
                    'id': tracker.id,
                    'bbox': [state[0], state[1], state[2], state[3]],
                    'center': (cx, cy)
                })
        
        return active_tracks