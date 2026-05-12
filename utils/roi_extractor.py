import os
import cv2
import argparse
import json

def load_roi_config(config_path):
    """
    加载ROI配置文件
    :param config_path: ROI配置文件路径（JSON格式）
    :return: ROI配置字典，如果文件不存在则返回None
    """
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return json.load(f)
    return None

def save_roi_config(config_path, roi_info):
    """
    保存ROI配置到JSON文件
    :param config_path: 输出配置文件路径
    :param roi_info: ROI配置信息字典
    """
    with open(config_path, 'w') as f:
        json.dump(roi_info, f, indent=2)

def resize_image_for_display(image, max_width=1280, max_height=720):
    """
    调整图像大小以适应显示窗口
    :param image: 输入图像
    :param max_width: 最大显示宽度
    :param max_height: 最大显示高度
    :return: 调整后的图像和缩放比例
    """
    height, width = image.shape[:2]
    
    # 计算缩放比例，保持宽高比
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
    """
    交互式选择ROI区域
    用户可以在图像上拖动鼠标选择第一视角和小地图两个ROI区域
    :param image: 输入图像
    :return: ROI列表，每个元素包含name、x、y、width、height
    """
    window_name = "Select ROI (Press Enter to confirm, Esc to cancel)"
    rois = []
    
    # 定义需要选择的ROI名称
    roi_names = ["first_person_view", "minimap"]
    
    # 调整图像大小以便显示
    resized_image, scale = resize_image_for_display(image)
    print(f"Image resized for display: scale={scale:.4f}")
    
    # 依次让用户选择每个ROI
    for roi_name in roi_names:
        cv2.imshow(window_name, resized_image)
        cv2.setWindowTitle(window_name, f"Select {roi_name} - Drag to select, Press Enter")
        
        # 使用OpenCV的selectROI函数进行交互式选择
        rect = cv2.selectROI(window_name, resized_image, fromCenter=False, showCrosshair=True)
        if rect == (0, 0, 0, 0):
            print(f"Skipped {roi_name}")
            rois.append(None)
        else:
            x, y, w, h = rect
            # 将坐标转换回原始图像的坐标系
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
    """
    从图像中提取指定的ROI区域
    :param image: 输入图像
    :param roi: ROI配置字典，包含x、y、width、height
    :return: 裁剪后的ROI图像
    """
    x, y, w, h = roi['x'], roi['y'], roi['width'], roi['height']
    return image[y:y+h, x:x+w]

def extract_rois_from_image(image, rois):
    """
    从图像中提取所有指定的ROI区域
    :param image: 输入图像
    :param rois: ROI配置列表
    :return: ROI图像字典 {roi_name: roi_image}
    """
    roi_images = {}
    for roi in rois:
        if roi is not None:
            roi_image = extract_roi(image, roi)
            roi_images[roi['name']] = roi_image
    return roi_images

def process_single_frame(frame_path, output_dir, rois, show_preview=False):
    """
    处理单帧图像，提取并保存所有ROI区域
    :param frame_path: 输入帧图像路径
    :param output_dir: 输出目录
    :param rois: ROI配置列表
    :param show_preview: 是否显示预览窗口
    """
    image = cv2.imread(frame_path)
    if image is None:
        print(f"Error: Could not read image {frame_path}")
        return
    
    frame_name = os.path.splitext(os.path.basename(frame_path))[0]
    
    # 提取所有ROI
    roi_images = extract_rois_from_image(image, rois)
    
    # 保存每个ROI图像
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
    """
    批量处理文件夹中的所有帧图像
    :param frames_dir: 包含帧图像的目录
    :param output_dir: 输出根目录
    :param rois: ROI配置列表
    :param show_preview: 是否显示预览窗口
    """
    if not os.path.exists(frames_dir):
        print(f"Error: Frames directory {frames_dir} does not exist")
        return
    
    # 查找所有图像文件
    image_files = [f for f in sorted(os.listdir(frames_dir)) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    
    if not image_files:
        print("No image files found in the frames directory")
        return
    
    print(f"Processing {len(image_files)} frames...")
    
    # 逐个处理帧图像
    for idx, image_file in enumerate(image_files):
        image_path = os.path.join(frames_dir, image_file)
        process_single_frame(image_path, output_dir, rois, show_preview)
        
        # 每处理100帧打印一次进度
        if (idx + 1) % 100 == 0:
            print(f"Processed {idx + 1}/{len(image_files)} frames")
    
    if show_preview:
        cv2.destroyAllWindows()
    
    print(f"Processing complete. Results saved to {output_dir}")

def process_videos_frames(video_names, frames_root_dir, output_root_dir, rois, show_preview=False):
    """
    处理多个视频的帧图像
    :param video_names: 视频名称列表
    :param frames_root_dir: 帧图像根目录
    :param output_root_dir: 输出根目录
    :param rois: ROI配置列表
    :param show_preview: 是否显示预览窗口
    """
    for video_name in video_names:
        frames_dir = os.path.join(frames_root_dir, video_name)
        output_dir = os.path.join(output_root_dir, video_name)
        
        print(f"\nProcessing video: {video_name}")
        process_frames_dir(frames_dir, output_dir, rois, show_preview)

def main():
    """
    主函数 - 解析命令行参数并执行ROI提取流程
    支持交互式选择ROI或从配置文件加载ROI
    """
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
    
    # 尝试加载现有的ROI配置
    rois = load_roi_config(args.config_path)
    
    # 如果是交互模式或没有配置文件，进入交互式选择
    if args.interactive or rois is None:
        sample_image = None
        # 确定样本图像路径
        if args.video_name:
            sample_frames_dir = os.path.join(args.frames_dir, args.video_name)
        else:
            video_dirs = [d for d in os.listdir(args.frames_dir) 
                         if os.path.isdir(os.path.join(args.frames_dir, d))]
            if video_dirs:
                sample_frames_dir = os.path.join(args.frames_dir, video_dirs[0])
            else:
                sample_frames_dir = args.frames_dir
        
        # 查找样本图像
        image_files = [f for f in os.listdir(sample_frames_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        if image_files:
            sample_image_path = os.path.join(sample_frames_dir, image_files[0])
            sample_image = cv2.imread(sample_image_path)
        
        if sample_image is not None:
            print("Please select ROIs interactively...")
            rois = select_roi_interactive(sample_image)
            
            # 如果选择了ROI，保存配置
            if any(rois):
                save_roi_config(args.config_path, rois)
                print(f"ROI config saved to {args.config_path}")
            else:
                print("No ROIs selected. Exiting.")
                return
        else:
            print("No sample image found for ROI selection.")
            return
    
    # 确定要处理的视频列表
    if args.video_name:
        video_names = [args.video_name]
    else:
        video_names = [d for d in os.listdir(args.frames_dir) 
                      if os.path.isdir(os.path.join(args.frames_dir, d))]
    
    if not video_names:
        video_names = ['']
    
    # 开始处理
    process_videos_frames(video_names, args.frames_dir, args.output_dir, rois, args.show_preview)

if __name__ == "__main__":
    main()