import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import pandas as pd

class Visualizer:
    def __init__(self):
        self.colors = [
            (255, 0, 0),    # red
            (0, 255, 0),    # green
            (0, 0, 255),    # blue
            (255, 255, 0),  # yellow
            (255, 0, 255),  # magenta
            (0, 255, 255),  # cyan
            (128, 0, 0),    # maroon
            (0, 128, 0),    # dark green
        ]
    
    def draw_roi(self, image, roi, alpha=0.3, label='ROI', color=(0, 255, 255)):
        """
        绘制半透明ROI区域（在最底层）
        :param image: 输入图像
        :param roi: ROI区域 (x1, y1, x2, y2)
        :param alpha: 透明度系数 (0-1)，值越小越透明
        :param label: ROI标签文字
        :param color: ROI颜色 (B, G, R)
        :return: 绘制后的图像
        """
        if roi is None:
            return image
        
        x1, y1, x2, y2 = roi
        # 创建半透明填充层
        overlay = image.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        # 设置透明度并混合
        cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
        # 绘制边框和文字
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
        cv2.putText(image, label, (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return image
    
    def draw_minimap_roi(self, image, roi, alpha=0.3):
        """
        绘制小地图ROI区域（黄色）
        注意：OpenCV使用BGR格式，黄色=(0, 255, 255)
        """
        return self.draw_roi(image, roi, alpha=alpha, label='ROI(minimap)', color=(0, 255, 255))
    
    def draw_first_person_roi(self, image, roi, alpha=0.3):
        """
        绘制第一视角ROI区域（淡珊瑚红）
        """
        return self.draw_roi(image, roi, alpha=alpha, label='ROI(first_person)', color=(100, 100, 180))
    
    def draw_tracks(self, image, tracks):
        for track in tracks:
            x1, y1, x2, y2 = track['bbox']
            cx, cy = track['center']
            color = self.colors[(track['id'] - 1) % len(self.colors)]
            
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.circle(image, (cx, cy), 4, color, -1)
            cv2.putText(image, str(track['id']), (cx + 5, cy - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return image
    
    def draw_trajectory(self, image, trajectory, color_idx=0, line_width=2):
        if len(trajectory) < 2:
            return image
        
        color = self.colors[color_idx % len(self.colors)]
        
        for i in range(1, len(trajectory)):
            prev_x, prev_y = trajectory[i-1]['x'], trajectory[i-1]['y']
            curr_x, curr_y = trajectory[i]['x'], trajectory[i]['y']
            
            cv2.line(image, (prev_x, prev_y), (curr_x, curr_y), color, line_width)
        
        return image
    
    def save_trajectory_map(self, trajectories, output_path, frame_shape=(480, 640, 3)):
        background = np.zeros(frame_shape, dtype=np.uint8)
        
        for idx, (track_id, trajectory) in enumerate(trajectories.items()):
            background = self.draw_trajectory(background, trajectory, idx)
        
        cv2.imwrite(output_path, background)
    
    def plot_trajectory_graph(self, trajectories, output_path):
        plt.figure(figsize=(12, 8))
        
        for idx, (track_id, trajectory) in enumerate(trajectories.items()):
            xs = [p['x'] for p in trajectory]
            ys = [p['y'] for p in trajectory]
            plt.plot(xs, ys, label=f'Player {track_id}', color=f'C{idx}', linewidth=2)
        
        plt.gca().invert_yaxis()
        plt.xlabel('X Pixel')
        plt.ylabel('Y Pixel')
        plt.title('Player Trajectories')
        plt.legend()
        plt.grid(True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def save_tracks_csv(self, trajectories, output_path):
        rows = []
        for track_id, trajectory in trajectories.items():
            for point in trajectory:
                rows.append({
                    'frame': point['frame_idx'],
                    'id': track_id,
                    'x': point['x'],
                    'y': point['y']
                })
        
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)
    
    def save_frame_with_tracks(self, frame, tracks, output_path):
        frame_with_tracks = frame.copy()
        frame_with_tracks = self.draw_tracks(frame_with_tracks, tracks)
        cv2.imwrite(output_path, frame_with_tracks)