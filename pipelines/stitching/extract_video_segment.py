#!/usr/bin/env python3
"""
Extract a time segment from video files.

Example: Extract minutes 3:30 to 5:30 from a video.
"""
import argparse
from pathlib import Path
import cv2


def extract_segment(input_video: Path, output_video: Path, start_time: str, end_time: str):
    """
    Extract a time segment from a video.
    
    Args:
        input_video: Path to input video
        output_video: Path to save extracted segment
        start_time: Start time in format "MM:SS" or "HH:MM:SS"
        end_time: End time in format "MM:SS" or "HH:MM:SS"
    """
    def time_to_seconds(time_str):
        """Convert time string to seconds."""
        parts = time_str.split(':')
        if len(parts) == 2:  # MM:SS
            return int(parts[0]) * 60 + int(parts[1])
        elif len(parts) == 3:  # HH:MM:SS
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
        else:
            raise ValueError(f"Invalid time format: {time_str}. Use MM:SS or HH:MM:SS")
    
    start_sec = time_to_seconds(start_time)
    end_sec = time_to_seconds(end_time)
    duration = end_sec - start_sec
    
    if duration <= 0:
        raise ValueError(f"End time must be after start time. Got {start_time} to {end_time}")
    
    print(f"Extracting segment from {input_video}")
    print(f"  Start: {start_time} ({start_sec}s)")
    print(f"  End: {end_time} ({end_sec}s)")
    print(f"  Duration: {duration}s")
    
    # Open input video
    cap = cv2.VideoCapture(str(input_video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {input_video}")
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_duration = total_frames / fps
    
    print(f"  Video properties: {width}x{height} @ {fps:.2f} fps, {total_duration:.1f}s total")
    
    # Validate time range
    if start_sec >= total_duration:
        raise ValueError(f"Start time {start_sec}s is beyond video duration {total_duration:.1f}s")
    if end_sec > total_duration:
        print(f"  ⚠️  Warning: End time {end_sec}s exceeds video duration {total_duration:.1f}s")
        print(f"     Adjusting to video end")
        end_sec = int(total_duration)
        duration = end_sec - start_sec
    
    # Seek to start position
    start_frame = int(start_sec * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    # Create output video writer
    output_video.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_video), fourcc, fps, (width, height))
    
    if not out.isOpened():
        raise RuntimeError(f"Could not create output video: {output_video}")
    
    # Extract frames
    frames_to_extract = int(duration * fps)
    frame_count = 0
    
    print(f"\nExtracting {frames_to_extract} frames...")
    
    while frame_count < frames_to_extract:
        ret, frame = cap.read()
        if not ret:
            print(f"  ⚠️  Reached end of video at frame {frame_count}")
            break
        
        out.write(frame)
        frame_count += 1
        
        if frame_count % (int(fps * 10)) == 0:  # Update every 10 seconds
            progress = (frame_count / frames_to_extract) * 100
            print(f"  Progress: {frame_count}/{frames_to_extract} frames ({progress:.1f}%)")
    
    cap.release()
    out.release()
    
    print(f"\n✅ Extracted segment saved to: {output_video}")
    print(f"   Frames extracted: {frame_count}")
    print(f"   Duration: {frame_count/fps:.1f}s")


def main():
    parser = argparse.ArgumentParser(
        description="Extract a time segment from video files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract 3:30 to 5:30 from a video
  python extract_video_segment.py \\
    --input data/cam_left_10min.mp4 \\
    --output data/cam_left_3_30_to_5_30.mp4 \\
    --start 3:30 \\
    --end 5:30

  # Extract from both videos
  python extract_video_segment.py \\
    --input data/cam_left_10min.mp4 \\
    --output data/cam_left_segment.mp4 \\
    --start 3:30 \\
    --end 5:30
  
  python extract_video_segment.py \\
    --input data/cam_right_10min.mp4 \\
    --output data/cam_right_segment.mp4 \\
    --start 3:30 \\
    --end 5:30
        """
    )
    
    parser.add_argument("--input", type=Path, required=True, help="Input video path")
    parser.add_argument("--output", type=Path, required=True, help="Output video path")
    parser.add_argument("--start", type=str, required=True, 
                       help="Start time in MM:SS or HH:MM:SS format (e.g., 3:30 or 0:03:30)")
    parser.add_argument("--end", type=str, required=True,
                       help="End time in MM:SS or HH:MM:SS format (e.g., 5:30 or 0:05:30)")
    
    args = parser.parse_args()
    
    if not args.input.exists():
        raise FileNotFoundError(f"Input video not found: {args.input}")
    
    extract_segment(args.input, args.output, args.start, args.end)


if __name__ == "__main__":
    main()

