import os
import cv2
import argparse

def extract_frames(video_path, output_dir, fps=None, start_frame=0, end_frame=None):
    os.makedirs(output_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return 0
    
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / video_fps
    
    if end_frame is None:
        end_frame = total_frames
    
    print(f"Video: {os.path.basename(video_path)}")
    print(f"FPS: {video_fps:.2f}")
    print(f"Total frames: {total_frames}")
    print(f"Duration: {duration:.2f} seconds")
    print(f"Extracting frames {start_frame} to {end_frame}")
    
    if fps is not None:
        frame_interval = int(video_fps / fps)
        print(f"Target FPS: {fps}, Frame interval: {frame_interval}")
    else:
        frame_interval = 1
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    saved_count = 0
    frame_idx = start_frame
    
    while frame_idx < end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        
        if (frame_idx - start_frame) % frame_interval == 0:
            frame_filename = f"frame_{frame_idx:06d}.jpg"
            frame_path = os.path.join(output_dir, frame_filename)
            cv2.imwrite(frame_path, frame)
            saved_count += 1
        
        frame_idx += 1
    
    cap.release()
    print(f"Successfully saved {saved_count} frames to {output_dir}")
    return saved_count

def process_videos_in_folder(videos_dir, output_dir, fps=None, start_frame=0, end_frame=None):
    if not os.path.exists(videos_dir):
        print(f"Error: Videos directory {videos_dir} does not exist")
        return
    
    video_files = [f for f in os.listdir(videos_dir) if f.endswith(('.mp4', '.avi', '.mov', '.mkv', '.flv'))]
    
    if not video_files:
        print("No video files found in the videos directory")
        return
    
    for video_file in video_files:
        video_path = os.path.join(videos_dir, video_file)
        video_name = os.path.splitext(video_file)[0]
        video_output_dir = os.path.join(output_dir, video_name)
        
        print(f"\nProcessing: {video_file}")
        extract_frames(video_path, video_output_dir, fps, start_frame, end_frame)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract frames from videos")
    parser.add_argument("--videos_dir", default="../videos", help="Directory containing video files")
    parser.add_argument("--output_dir", default="../datasets/frames", help="Directory to save extracted frames")
    parser.add_argument("--fps", type=float, default=None, help="Target FPS for frame extraction (default: all frames)")
    parser.add_argument("--start_frame", type=int, default=0, help="Start frame index")
    parser.add_argument("--end_frame", type=int, default=None, help="End frame index")
    args = parser.parse_args()
    
    process_videos_in_folder(args.videos_dir, args.output_dir, args.fps, args.start_frame, args.end_frame)