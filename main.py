import cv2
import os
import json
import numpy as np
import pandas as pd
from detector import PlayerDetector
from tracker import TrackerManager
from trajectory import TrajectoryExtractor

class EventCollector:
    def __init__(self):
        self.events = []
    
    def add_event(self, frame_idx, event_type, player_id=None, details=None):
        event = {
            'frame_idx': frame_idx,
            'event_type': event_type,
            'player_id': player_id
        }
        if details:
            event.update(details)
        self.events.append(event)
    
    def save_csv(self, output_path, fps=30):
        df = pd.DataFrame(self.events)
        df['timestamp'] = df['frame_idx'] / fps
        df.to_csv(output_path, index=False)
        return df

def process_video(video_path, output_dir, map_roi=None, conf_threshold=0.5):
    os.makedirs(output_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Processing video: {video_path}")
    print(f"FPS: {fps:.2f}, Total frames: {total_frames}, Resolution: {width}x{height}")
    print(f"Map ROI: {map_roi}")
    print(f"Confidence threshold: {conf_threshold}")
    
    detector = PlayerDetector(model_path='weights/best.pt', conf_threshold=conf_threshold)
    tracker_manager = TrackerManager()
    trajectory_extractor = TrajectoryExtractor()
    event_collector = EventCollector()
    
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    video_output_path = os.path.join(output_dir, f'{video_name}_tracking.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(video_output_path, fourcc, fps, (width, height))
    
    frame_idx = 0
    output_frame_dir = os.path.join(output_dir, 'frames')
    os.makedirs(output_frame_dir, exist_ok=True)
    
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        detections = detector.detect(frame, roi=map_roi)
        
        tracks = tracker_manager.update(detections)
        
        trajectory_extractor.update(tracks, frame_idx)
        
        frame_with_tracks = frame.copy()
        
        if map_roi is not None:
            x1, y1, x2, y2 = map_roi
            cv2.rectangle(frame_with_tracks, (x1, y1), (x2, y2), (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame_with_tracks, 'ROI(minimap)', (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        for track in tracks:
            x1, y1, x2, y2 = track['bbox']
            color = colors[(track['id'] - 1) % len(colors)]
            cv2.rectangle(frame_with_tracks, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame_with_tracks, str(track['id']), 
                       (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        trajectories = trajectory_extractor.get_all_trajectories()
        for idx, (track_id, traj) in enumerate(trajectories.items()):
            color = colors[idx % len(colors)]
            for i in range(1, len(traj)):
                cv2.line(frame_with_tracks, 
                        (traj[i-1]['x'], traj[i-1]['y']),
                        (traj[i]['x'], traj[i]['y']),
                        color, 2)
        
        video_writer.write(frame_with_tracks)
        
        if frame_idx % 30 == 0:
            frame_output_path = os.path.join(output_frame_dir, f"frame_{frame_idx:06d}.jpg")
            cv2.imwrite(frame_output_path, frame_with_tracks)
        
        if frame_idx % 100 == 0:
            print(f"Processed frame {frame_idx}/{total_frames}")
        
        frame_idx += 1
    
    cap.release()
    video_writer.release()
    
    trajectories = trajectory_extractor.get_all_trajectories()
    
    for track_id, traj in trajectories.items():
        traj_path = os.path.join(output_dir, f'{video_name}_track_{track_id}.json')
        with open(traj_path, 'w') as f:
            json.dump(traj, f, indent=2)
        
        direction_changes = trajectory_extractor.detect_direction_change(track_id)
        for change in direction_changes:
            event_collector.add_event(
                frame_idx=change['frame_idx'],
                event_type='direction_change',
                player_id=track_id,
                details={'angle': change['angle'], 'position': change['position']}
            )
    
    tracks_csv_path = os.path.join(output_dir, f'{video_name}_tracks.csv')
    rows = []
    for track_id, traj in trajectories.items():
        for point in traj:
            rows.append({
                'frame': point['frame_idx'],
                'id': track_id,
                'x': point['x'],
                'y': point['y']
            })
    df = pd.DataFrame(rows)
    df.to_csv(tracks_csv_path, index=False)
    
    events_csv_path = os.path.join(output_dir, f'{video_name}_events.csv')
    event_collector.save_csv(events_csv_path, fps=fps)
    
    traj_map_path = os.path.join(output_dir, f'{video_name}_trajectory_map.jpg')
    traj_map = np.zeros((height, width, 3), dtype=np.uint8)
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
    for idx, (track_id, traj) in enumerate(trajectories.items()):
        color = colors[idx % len(colors)]
        for i in range(1, len(traj)):
            cv2.line(traj_map, 
                    (traj[i-1]['x'], traj[i-1]['y']),
                    (traj[i]['x'], traj[i]['y']),
                    color, 2)
    cv2.imwrite(traj_map_path, traj_map)
    
    print(f"\nProcessing complete!")
    print(f"Results saved to: {output_dir}")
    print(f"Tracked {len(trajectories)} players")
    print(f"Detected {len(event_collector.events)} direction changes")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="FPS游戏选手轨迹提取")
    parser.add_argument("--video", required=True, help="视频文件路径")
    parser.add_argument("--output", default="outputs", help="输出目录")
    parser.add_argument("--map_roi", type=str, default=None, 
                        help="地图ROI区域 (格式: x1,y1,x2,y2)")
    parser.add_argument("--conf_threshold", type=float, default=0.5, 
                        help="检测置信度阈值")
    args = parser.parse_args()
    
    map_roi = None
    if args.map_roi:
        coords = list(map(int, args.map_roi.split(',')))
        if len(coords) == 4:
            map_roi = tuple(coords)
            print(f"Using map ROI: {map_roi}")
    
    process_video(args.video, args.output, map_roi, args.conf_threshold)

if __name__ == "__main__":
    main()