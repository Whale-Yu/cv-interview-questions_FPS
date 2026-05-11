import numpy as np
from filterpy.kalman import KalmanFilter

class KalmanBoxTracker:
    def __init__(self, bbox):
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([[1,0,0,0,1,0,0],
                              [0,1,0,0,0,1,0],
                              [0,0,1,0,0,0,1],
                              [0,0,0,1,0,0,0],
                              [0,0,0,0,1,0,0],
                              [0,0,0,0,0,1,0],
                              [0,0,0,0,0,0,1]])
        self.kf.H = np.array([[1,0,0,0,0,0,0],
                              [0,1,0,0,0,0,0],
                              [0,0,1,0,0,0,0],
                              [0,0,0,1,0,0,0]])
        self.kf.P[4:,4:] *= 1000.0
        self.kf.P *= 10.0
        self.kf.R = np.array([[10, 0, 0, 0],
                             [0, 10, 0, 0],
                             [0, 0, 10, 0],
                             [0, 0, 0, 10]])
        self.kf.Q[-1,-1] *= 0.01
        self.kf.Q[4:,4:] *= 0.01
        
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        self.kf.x[:4] = np.array([[x1], [y1], [w], [h]])
        self.id = None
        self.age = 0
        self.hits = 0
        self.time_since_update = 0
    
    def update(self, bbox):
        self.time_since_update = 0
        self.hits += 1
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        self.kf.update(np.array([[x1], [y1], [w], [h]]))
    
    def predict(self):
        if self.kf.x[6] + self.kf.x[2] <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hits = 0
        self.time_since_update += 1
        return self.get_state()
    
    def get_state(self):
        x = self.kf.x[:4].flatten()
        return [int(x[0]), int(x[1]), int(x[0]+x[2]), int(x[1]+x[3])]

class TrackerManager:
    def __init__(self, max_age=30, min_hits=1):
        self.trackers = []
        self.max_age = max_age
        self.min_hits = min_hits
        self.next_id = 1
    
    def iou(self, boxA, boxB):
        x1, y1, x2, y2 = boxA
        x3, y3, x4, y4 = boxB
        
        inter_x1 = max(x1, x3)
        inter_y1 = max(y1, y3)
        inter_x2 = min(x2, x4)
        inter_y2 = min(y2, y4)
        
        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
        areaA = (x2 - x1) * (y2 - y1)
        areaB = (x4 - x3) * (y4 - y3)
        
        return inter_area / float(areaA + areaB - inter_area) if (areaA + areaB - inter_area) > 0 else 0
    
    def update(self, detections):
        for tracker in self.trackers:
            tracker.predict()
        
        matched = []
        unmatched_detections = list(range(len(detections)))
        unmatched_trackers = list(range(len(self.trackers)))
        
        for t_idx, tracker in enumerate(self.trackers):
            best_iou = 0
            best_d_idx = -1
            
            for d_idx in unmatched_detections:
                iou = self.iou(tracker.get_state(), detections[d_idx]['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_d_idx = d_idx
            
            if best_iou > 0.3 and best_d_idx != -1:
                tracker.update(detections[best_d_idx]['bbox'])
                matched.append((t_idx, best_d_idx))
                unmatched_detections.remove(best_d_idx)
                unmatched_trackers.remove(t_idx)
        
        for d_idx in unmatched_detections:
            new_tracker = KalmanBoxTracker(detections[d_idx]['bbox'])
            new_tracker.id = self.next_id
            self.next_id += 1
            self.trackers.append(new_tracker)
        
        self.trackers = [t for t in self.trackers if t.time_since_update <= self.max_age]
        
        active_tracks = []
        for tracker in self.trackers:
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