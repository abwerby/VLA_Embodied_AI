import cv2
import os
from pathlib import Path
from tqdm import tqdm
import argparse

def extract_frames(video_path, save_path, interval_seconds):
    """
    Extract frames from a video file at specified intervals
    
    Args:
        video_path (str): Path to the input video file
        save_path (str): Directory to save extracted frames
        interval_seconds (float): Interval between frames in seconds
        
    Returns:
        int: Number of frames extracted
    """
    # Validate inputs
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    # Create save directory if it doesn't exist
    Path(save_path).mkdir(parents=True, exist_ok=True)
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Error opening video file: {video_path}")
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * interval_seconds)
    
    # Extract frames
    frame_count = 0
    current_frame = 0
    
    for i in tqdm(range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))):
        ret, frame = cap.read()
        if not ret:
            break
        
        if current_frame % frame_interval == 0:
            frame_path = os.path.join(save_path, f"frame_{frame_count:04d}.jpg")
            cv2.imwrite(frame_path, frame)
            frame_count += 1
        
        current_frame += 1
    
    # Release resources
    cap.release()
    
    return frame_count

def main():
    parser = argparse.ArgumentParser(description="Extract frames from a video file at specified intervals.")
    parser.add_argument("video_path", type=str, help="Path to the input video file")
    parser.add_argument("--interval", type=float, default=1.0, help="Interval between frames in seconds")
    args = parser.parse_args()

    video_path = args.video_path
    interval_seconds = args.interval

    # Create frames directory using video filename
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    save_path = os.path.join(os.path.dirname(video_path), f"{video_name}_frames")
    
    try:
        num_frames = extract_frames(video_path, save_path, interval_seconds)
        print(f"Successfully extracted {num_frames} frames to {save_path}")
    except FileNotFoundError as fnf_error:
        print(f"File not found error: {str(fnf_error)}")
    except RuntimeError as rt_error:
        print(f"Runtime error: {str(rt_error)}")
    except Exception as e:
        print(f"Error processing video: {str(e)}")


if __name__ == "__main__":
    main()