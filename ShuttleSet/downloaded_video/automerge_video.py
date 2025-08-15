#!/usr/bin/env python3
import os
import sys
import subprocess
import re
import argparse

# Assuming the script is run from the directory containing the raw videos
source_video_dir = '.' 
ffmpeg_path = '/opt/homebrew/bin/ffmpeg'
ffprobe_path = '/opt/homebrew/bin/ffprobe'

def find_media_streams(files):
    """
    Inspects a list of files and identifies the best video and an audio file.
    """
    video_file = None
    audio_file = None
    max_resolution = 0

    for f in files:
        filepath = os.path.join(source_video_dir, f)
        # print(f"Processing file: {f}")
        # Check for video streams and find the highest resolution one
        try:
            command = [
                ffprobe_path, '-v', 'error', '-select_streams', 'v:0',
                '-show_entries', 'stream=width,height', '-of', 'csv=s=x:p=0', f"{filepath}"
            ]
            # print(f"Running command: {' '.join(command)}")
            result = subprocess.run(
                command,
                capture_output=True, text=True, check=False
            )
            if result.returncode == 0 and result.stdout.strip():
                output = result.stdout.strip().split('\n')[0]
                width, height = map(int, output.split('x'))
                if width * height > max_resolution:
                    max_resolution = width * height
                    print(f"New best video found: {f} with resolution {width}x{height}")
                    video_file = f
        except Exception:
            pass # Not a video file or error probing

        # Check for audio streams
        try:
            command = [
                ffprobe_path, '-v', 'error', '-select_streams', 'a:0',
                '-show_entries', 'stream=codec_type', '-of', 'csv=p=0', f"{filepath}"
            ]
            result = subprocess.run(
                command,
                capture_output=True, text=True, check=False
            )
            if result.returncode == 0 and result.stdout.strip() == 'audio':
                print(f"New best audio found: {f}")
                audio_file = f
        except Exception:
            pass # Not an audio file or error probing

    if audio_file == video_file and len(files) > 1:
        for f in files:
            if f != video_file:
                audio_file = f
                break

    return video_file, audio_file

def main():
    parser = argparse.ArgumentParser(description="Merge video and audio files based on a common base name.")
    parser.add_argument("filename", help="A representative filename (video or audio part).")
    parser.add_argument("-d", "--destination", default="../raw_video", help="The destination directory for the merged video. Defaults to './combined'.")
    args = parser.parse_args()

    input_filename = args.filename
    destination_dir = args.destination

    if not os.path.exists(os.path.join(source_video_dir, input_filename)):
        print(f"Error: File not found in the current directory: {input_filename}")
        sys.exit(1)
    print(f"Path exists: {input_filename}")
    base_name = re.sub(r'\.(mp4|mkv|avi|mov|flv|webm)$', '', input_filename)
    base_name = re.sub(r'\.(f\d+)', '', base_name).strip()
    print(f"Detected base name: {base_name}")

    related_files = [f for f in os.listdir(source_video_dir) if f.startswith(base_name) and os.path.isfile(os.path.join(source_video_dir, f))]
    if len(related_files) < 2:
        print("Could not find a pair of files to merge.")
        sys.exit(0)
    
    print(f"Found related files: {', '.join(related_files)}")

    video_file, audio_file = find_media_streams(related_files)

    if not video_file or not audio_file:
        print("Error: Could not identify a clear video and audio pair.")
        sys.exit(1)

    print(f"Identified Video File: {video_file}")
    print(f"Identified Audio File: {audio_file}")

    os.makedirs(destination_dir, exist_ok=True)
    output_filename = f"{base_name}.mp4"
    output_path = os.path.join(destination_dir, output_filename)

    if os.path.exists(output_path):
        print(f"Output file '{output_filename}' already exists in '{destination_dir}'. Skipping.")
        sys.exit(0)

    video_path = os.path.join(source_video_dir, video_file)
    audio_path = os.path.join(source_video_dir, audio_file)

    print(f"Merging into '{output_path}'...")
    try:
        subprocess.run(
            [
                ffmpeg_path,
                '-i', video_path,
                '-i', audio_path,
                '-c:v', 'copy',
                '-c:a', 'aac',
                '-strict', 'experimental',
                output_path
            ],
            check=True,
            capture_output=True,
        )
        print("Successfully merged video and audio.")
    except subprocess.CalledProcessError as e:
        print("Error during merging process:")
        print(e.stderr.decode())

if __name__ == "__main__":
    main()
