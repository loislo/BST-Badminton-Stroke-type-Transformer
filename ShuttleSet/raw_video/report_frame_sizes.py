#!/usr/bin/env python3
import os
import subprocess

ffprobe_path = '/opt/homebrew/bin/ffprobe'
current_dir = '.'

# Sort the directory listing to ensure a consistent order
for filename in sorted(os.listdir(current_dir)):
    filepath = os.path.join(current_dir, filename)

    # Check if it's a file, not a directory
    if os.path.isfile(filepath):
        try:
            # Run ffprobe to get the width and height of the video stream
            result = subprocess.run(
                [
                    ffprobe_path,
                    '-v',
                    'error',
                    '-select_streams',
                    'v:0',
                    '-show_entries',
                    'stream=width,height',
                    '-of',
                    'csv=s=x:p=0',
                    filepath
                ],
                capture_output=True,
                text=True,
                check=False # Don't raise an exception for non-video files
            )

            # If ffprobe ran successfully and produced output, process it
            if result.returncode == 0 and result.stdout.strip():
                dimensions = result.stdout.strip().split('\n')[0]
                if 'x' in dimensions:
                    width, height = dimensions.split('x')
                    print(f"{width}x{height} - {filename}")

        except Exception as e:
            # This will catch other errors, e.g., if ffprobe can't be found
            print(f"Error processing {filename}: {e}")
