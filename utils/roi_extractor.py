import os
import cv2
import argparse
import json

def load_roi_config(config_path):
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return json.load(f)
    return None

def save_roi_config(config_path, roi_info):
    with open(config_path, 'w') as f:
        json.dump(roi_info, f, indent=2)

def resize_image_for_display(image, max_width=1280, max_height=720):
    height, width = image.shape[:2]
    
    scale = 1.0
    if width > max_width or height > max_height:
        scale_width = max_width / width
        scale_height = max_height / height
        scale = min(scale_width, scale_height)
    
    new_width = int(width * scale)
    new_height = int(height * scale)
    
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return resized, scale

def select_roi_interactive(image):
    window_name = "Select ROI (Press Enter to confirm, Esc to cancel)"
    rois = []
    
    roi_names = ["first_person_view", "minimap"]
    
    resized_image, scale = resize_image_for_display(image)
    print(f"Image resized for display: scale={scale:.4f}")
    
    for roi_name in roi_names:
        cv2.imshow(window_name, resized_image)
        cv2.setWindowTitle(window_name, f"Select {roi_name} - Drag to select, Press Enter")
        
        rect = cv2.selectROI(window_name, resized_image, fromCenter=False, showCrosshair=True)
        if rect == (0, 0, 0, 0):
            print(f"Skipped {roi_name}")
            rois.append(None)
        else:
            x, y, w, h = rect
            original_x = int(x / scale)
            original_y = int(y / scale)
            original_w = int(w / scale)
            original_h = int(h / scale)
            
            rois.append({
                'name': roi_name,
                'x': original_x,
                'y': original_y,
                'width': original_w,
                'height': original_h
            })
            print(f"Selected {roi_name}: x={original_x}, y={original_y}, w={original_w}, h={original_h} (on scaled image: x={x}, y={y}, w={w}, h={h})")
    
    cv2.destroyAllWindows()
    return rois

def extract_roi(image, roi):
    x, y, w, h = roi['x'], roi['y'], roi['width'], roi['height']
    return image[y:y+h, x:x+w]

def extract_rois_from_image(image, rois):
    roi_images = {}
    for roi in rois:
        if roi is not None:
            roi_image = extract_roi(image, roi)
            roi_images[roi['name']] = roi_image
    return roi_images

def process_single_frame(frame_path, output_dir, rois, show_preview=False):
    image = cv2.imread(frame_path)
    if image is None:
        print(f"Error: Could not read image {frame_path}")
        return
    
    frame_name = os.path.splitext(os.path.basename(frame_path))[0]
    
    roi_images = extract_rois_from_image(image, rois)
    
    for roi_name, roi_image in roi_images.items():
        roi_output_dir = os.path.join(output_dir, roi_name)
        os.makedirs(roi_output_dir, exist_ok=True)
        
        output_path = os.path.join(roi_output_dir, f"{frame_name}_{roi_name}.jpg")
        cv2.imwrite(output_path, roi_image)
        
        if show_preview:
            cv2.imshow(f"ROI - {roi_name}", roi_image)
    
    if show_preview:
        cv2.waitKey(1)

def process_frames_dir(frames_dir, output_dir, rois, show_preview=False):
    if not os.path.exists(frames_dir):
        print(f"Error: Frames directory {frames_dir} does not exist")
        return
    
    image_files = [f for f in sorted(os.listdir(frames_dir)) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    
    if not image_files:
        print("No image files found in the frames directory")
        return
    
    print(f"Processing {len(image_files)} frames...")
    
    for idx, image_file in enumerate(image_files):
        image_path = os.path.join(frames_dir, image_file)
        process_single_frame(image_path, output_dir, rois, show_preview)
        
        if (idx + 1) % 100 == 0:
            print(f"Processed {idx + 1}/{len(image_files)} frames")
    
    if show_preview:
        cv2.destroyAllWindows()
    
    print(f"Processing complete. Results saved to {output_dir}")

def process_videos_frames(video_names, frames_root_dir, output_root_dir, rois, show_preview=False):
    for video_name in video_names:
        frames_dir = os.path.join(frames_root_dir, video_name)
        output_dir = os.path.join(output_root_dir, video_name)
        
        print(f"\nProcessing video: {video_name}")
        process_frames_dir(frames_dir, output_dir, rois, show_preview)

def main():
    parser = argparse.ArgumentParser(description="Extract ROIs from video frames")
    parser.add_argument("--frames_dir", default="../datasets/images", 
                        help="Directory containing extracted frames")
    parser.add_argument("--output_dir", default="../datasets/rois", 
                        help="Directory to save extracted ROIs")
    parser.add_argument("--config_path", default="roi_config.json", 
                        help="Path to ROI configuration file")
    parser.add_argument("--video_name", default=None, 
                        help="Process specific video (default: all videos in frames_dir)")
    parser.add_argument("--interactive", action="store_true", 
                        help="Interactive ROI selection mode")
    parser.add_argument("--show_preview", action="store_true", 
                        help="Show preview of extracted ROIs")
    args = parser.parse_args()
    
    rois = load_roi_config(args.config_path)
    
    if args.interactive or rois is None:
        sample_image = None
        if args.video_name:
            sample_frames_dir = os.path.join(args.frames_dir, args.video_name)
        else:
            video_dirs = [d for d in os.listdir(args.frames_dir) 
                         if os.path.isdir(os.path.join(args.frames_dir, d))]
            if video_dirs:
                sample_frames_dir = os.path.join(args.frames_dir, video_dirs[0])
            else:
                sample_frames_dir = args.frames_dir
        
        image_files = [f for f in os.listdir(sample_frames_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        if image_files:
            sample_image_path = os.path.join(sample_frames_dir, image_files[0])
            sample_image = cv2.imread(sample_image_path)
        
        if sample_image is not None:
            print("Please select ROIs interactively...")
            rois = select_roi_interactive(sample_image)
            
            if any(rois):
                save_roi_config(args.config_path, rois)
                print(f"ROI config saved to {args.config_path}")
            else:
                print("No ROIs selected. Exiting.")
                return
        else:
            print("No sample image found for ROI selection.")
            return
    
    if args.video_name:
        video_names = [args.video_name]
    else:
        video_names = [d for d in os.listdir(args.frames_dir) 
                      if os.path.isdir(os.path.join(args.frames_dir, d))]
    
    if not video_names:
        video_names = ['']
    
    process_videos_frames(video_names, args.frames_dir, args.output_dir, rois, args.show_preview)

if __name__ == "__main__":
    main()