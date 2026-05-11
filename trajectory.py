import numpy as np
from filterpy.kalman import KalmanFilter

class KalmanSmoother:
    def __init__(self):
        self.kf = KalmanFilter(dim_x=4, dim_z=2)
        self.kf.F = np.array([[1, 0, 1, 0],
                              [0, 1, 0, 1],
                              [0, 0, 1, 0],
                              [0, 0, 0, 1]])
        self.kf.H = np.array([[1, 0, 0, 0],
                              [0, 1, 0, 0]])
        self.kf.P *= 10.0
        self.kf.R = np.array([[1, 0], [0, 1]]) * 5
        self.first_update = True
    
    def update(self, x, y):
        if self.first_update:
            self.kf.x[:2] = np.array([[x], [y]])
            self.first_update = False
            return x, y
        
        self.kf.predict()
        self.kf.update(np.array([[x], [y]]))
        state = self.kf.x[:2].flatten()
        return int(state[0]), int(state[1])

class TrajectoryExtractor:
    def __init__(self):
        self.trajectories = {}
        self.smoothers = {}
    
    def update(self, tracks, frame_idx):
        for track in tracks:
            track_id = track['id']
            
            if track_id not in self.trajectories:
                self.trajectories[track_id] = []
                self.smoothers[track_id] = KalmanSmoother()
            
            x, y = track['center']
            smoothed_x, smoothed_y = self.smoothers[track_id].update(x, y)
            
            self.trajectories[track_id].append({
                'frame_idx': frame_idx,
                'x': smoothed_x,
                'y': smoothed_y,
                'raw_x': x,
                'raw_y': y,
                'bbox': track['bbox']
            })
    
    def get_trajectory(self, track_id):
        return self.trajectories.get(track_id, [])
    
    def get_all_trajectories(self):
        return self.trajectories
    
    def detect_direction_change(self, track_id, threshold_deg=30):
        traj = self.trajectories.get(track_id, [])
        if len(traj) < 3:
            return []
        
        changes = []
        for i in range(2, len(traj)):
            x0, y0 = traj[i-2]['x'], traj[i-2]['y']
            x1, y1 = traj[i-1]['x'], traj[i-1]['y']
            x2, y2 = traj[i]['x'], traj[i]['y']
            
            dx1 = x1 - x0
            dy1 = y1 - y0
            dx2 = x2 - x1
            dy2 = y2 - y1
            
            mag1 = np.sqrt(dx1**2 + dy1**2)
            mag2 = np.sqrt(dx2**2 + dy2**2)
            
            if mag1 > 0 and mag2 > 0:
                dot = dx1 * dx2 + dy1 * dy2
                cos_angle = dot / (mag1 * mag2)
                cos_angle = np.clip(cos_angle, -1, 1)
                angle = np.arccos(cos_angle) * 180 / np.pi
                
                if angle > threshold_deg:
                    changes.append({
                        'frame_idx': traj[i]['frame_idx'],
                        'angle': float(angle),
                        'position': (traj[i]['x'], traj[i]['y'])
                    })
        
        return changes