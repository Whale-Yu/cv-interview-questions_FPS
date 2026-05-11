import time
import argparse
import os
from datetime import datetime

def create_session_log():
    session_dir = os.path.join(os.path.dirname(__file__), 'sessions')
    os.makedirs(session_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(session_dir, f"session_{timestamp}.log")
    
    return log_file

def log_action(log_file, action, details=""):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(f"[{timestamp}] {action}: {details}\n")

def main():
    parser = argparse.ArgumentParser(description="Roboflow智能标注自动化工具")
    parser.add_argument("--wait_time", type=int, default=5, help="每张图片等待时间(秒)")
    parser.add_argument("--max_images", type=int, default=500, help="最大处理图片数量")
    parser.add_argument("--start_delay", type=int, default=10, help="启动后等待时间(秒)，用于切换到浏览器")
    args = parser.parse_args()
    
    log_file = create_session_log()
    print(f"会话日志已创建: {log_file}")
    log_action(log_file, "SESSION_START", f"wait_time={args.wait_time}s, max_images={args.max_images}")
    
    print(f"\n准备开始自动标注...")
    print(f"1. 请在{args.start_delay}秒内切换到Roboflow标注页面")
    print(f"2. 确保浏览器窗口处于活动状态")
    print(f"3. 程序将自动按右箭头切换图片")
    print(f"4. 每张图片等待{args.wait_time}秒")
    print(f"5. 总共处理{args.max_images}张图片")
    print(f"\n按 Ctrl+C 终止程序")
    
    for i in range(args.start_delay, 0, -1):
        print(f"\r倒计时: {i}秒", end="")
        time.sleep(1)
    
    print("\n开始自动切换...")
    log_action(log_file, "AUTO_START", "开始自动切换图片")
    
    try:
        import pyautogui
        pyautogui.FAILSAFE = True
        
        for image_idx in range(1, args.max_images + 1):
            print(f"\r处理第 {image_idx}/{args.max_images} 张图片...", end="")
            
            try:
                pyautogui.press('right')
                log_action(log_file, "IMAGE_SWITCH", f"第{image_idx}张")
                
                time.sleep(args.wait_time)
                
            except Exception as e:
                log_action(log_file, "ERROR", str(e))
                print(f"\n错误: {e}")
                break
        
        print(f"\n\n完成! 共处理{args.max_images}张图片")
        log_action(log_file, "SESSION_END", f"成功处理{args.max_images}张图片")
        
    except ImportError:
        print("\n错误: 需要安装 pyautogui 库")
        print("请运行: pip install pyautogui")
        log_action(log_file, "ERROR", "pyautogui未安装")
        
    except KeyboardInterrupt:
        print(f"\n\n用户终止程序")
        log_action(log_file, "SESSION_END", "用户手动终止")

if __name__ == "__main__":
    main()