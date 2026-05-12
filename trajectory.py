import numpy as np
from filterpy.kalman import KalmanFilter

class KalmanSmoother:
    """
    卡尔曼平滑器 - 用于平滑轨迹点，减少抖动
    使用4维状态空间：[x, y, vx, vy]，其中x,y是位置，vx,vy是速度
    """
    def __init__(self):
        """
        初始化卡尔曼平滑器
        设置状态转移矩阵、观测矩阵和噪声协方差等参数
        """
        # 创建4维状态的卡尔曼滤波器（位置+速度）
        self.kf = KalmanFilter(dim_x=4, dim_z=2)
        
        # 状态转移矩阵 F：描述位置和速度如何随时间变化
        self.kf.F = np.array([[1, 0, 1, 0],
                              [0, 1, 0, 1],
                              [0, 0, 1, 0],
                              [0, 0, 0, 1]])
        
        # 观测矩阵 H：只观测位置（x, y）
        self.kf.H = np.array([[1, 0, 0, 0],
                              [0, 1, 0, 0]])
        
        # 初始化协方差矩阵
        self.kf.P *= 10.0
        
        # 测量噪声协方差（假设测量误差较小）
        self.kf.R = np.array([[1, 0], [0, 1]]) * 5
        
        self.first_update = True  # 标记是否为第一次更新
    
    def update(self, x, y):
        """
        使用新的观测点更新卡尔曼滤波器并返回平滑后的坐标
        :param x: 观测点的x坐标
        :param y: 观测点的y坐标
        :return: 平滑后的(x, y)坐标
        """
        # 第一次更新时，直接初始化状态，不进行预测
        if self.first_update:
            self.kf.x[:2] = np.array([[x], [y]])
            self.first_update = False
            return x, y
        
        # 预测下一步状态
        self.kf.predict()
        # 使用观测值更新状态
        self.kf.update(np.array([[x], [y]]))
        # 返回平滑后的位置
        state = self.kf.x[:2].flatten()
        return int(state[0]), int(state[1])

class TrajectoryExtractor:
    """
    轨迹提取器 - 管理和提取玩家的运动轨迹
    使用卡尔曼平滑器对轨迹进行平滑处理，并支持检测方向变化
    """
    def __init__(self):
        """
        初始化轨迹提取器
        """
        self.trajectories = {}  # 存储所有轨迹数据 {track_id: [points]}
        self.smoothers = {}  # 为每个跟踪ID维护一个卡尔曼平滑器
    
    def update(self, tracks, frame_idx):
        """
        更新轨迹数据，为每个跟踪目标添加新的轨迹点
        :param tracks: 当前帧的跟踪结果列表
        :param frame_idx: 当前帧索引
        """
        for track in tracks:
            track_id = track['id']
            
            # 如果是新的跟踪ID，初始化轨迹和平滑器
            if track_id not in self.trajectories:
                self.trajectories[track_id] = []
                self.smoothers[track_id] = KalmanSmoother()
            
            # 获取原始坐标并进行平滑处理
            x, y = track['center']
            smoothed_x, smoothed_y = self.smoothers[track_id].update(x, y)
            
            # 保存轨迹点（包含原始坐标和平滑后的坐标）
            self.trajectories[track_id].append({
                'frame_idx': frame_idx,
                'x': smoothed_x,
                'y': smoothed_y,
                'raw_x': x,
                'raw_y': y,
                'bbox': track['bbox']
            })
    
    def get_trajectory(self, track_id):
        """
        获取指定跟踪ID的轨迹
        :param track_id: 跟踪ID
        :return: 轨迹点列表
        """
        return self.trajectories.get(track_id, [])
    
    def get_all_trajectories(self):
        """
        获取所有轨迹数据
        :return: 所有轨迹字典 {track_id: trajectory}
        """
        return self.trajectories
    
    def detect_direction_change(self, track_id, threshold_deg=30):
        """
        检测轨迹中的方向变化事件
        通过计算连续三点之间的夹角来判断是否发生显著转向
        :param track_id: 跟踪ID
        :param threshold_deg: 方向变化的角度阈值（度），默认30度
        :return: 方向变化事件列表
        """
        traj = self.trajectories.get(track_id, [])
        if len(traj) < 3:
            return []  # 至少需要3个点才能计算角度
        
        changes = []
        # 遍历轨迹，检查每三个连续点
        for i in range(2, len(traj)):
            # 获取三个连续点的坐标
            x0, y0 = traj[i-2]['x'], traj[i-2]['y']
            x1, y1 = traj[i-1]['x'], traj[i-1]['y']
            x2, y2 = traj[i]['x'], traj[i]['y']
            
            # 计算两个向量：P0->P1 和 P1->P2
            dx1 = x1 - x0
            dy1 = y1 - y0
            dx2 = x2 - x1
            dy2 = y2 - y1
            
            # 计算向量长度
            mag1 = np.sqrt(dx1**2 + dy1**2)
            mag2 = np.sqrt(dx2**2 + dy2**2)
            
            # 如果两个向量都有长度，计算夹角
            if mag1 > 0 and mag2 > 0:
                # 使用点积公式计算夹角余弦值
                dot = dx1 * dx2 + dy1 * dy2
                cos_angle = dot / (mag1 * mag2)
                # 限制在[-1, 1]范围内，避免数值误差
                cos_angle = np.clip(cos_angle, -1, 1)
                # 计算角度（转换为度）
                angle = np.arccos(cos_angle) * 180 / np.pi
                
                # 如果角度超过阈值，记录方向变化事件
                if angle > threshold_deg:
                    changes.append({
                        'frame_idx': traj[i]['frame_idx'],
                        'angle': float(angle),
                        'position': (traj[i]['x'], traj[i]['y'])
                    })
        
        return changes