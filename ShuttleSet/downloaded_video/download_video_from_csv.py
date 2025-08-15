#!/usr/bin/env python3
"""
download_video_from_csv.py

This script reads a CSV file and, for each URL, downloads the video using yt-dlp.
The final video is named using the pattern '<video_id> - <title>.mp4' and
is moved to a specified final destination directory. It checks if the file
already exists in the final directory before downloading.
"""

import csv
import pathlib
import sys
import argparse
import subprocess
from typing import List, Dict, Any
from urllib.parse import urlparse, parse_qs
import shutil


def read_video_data(csv_path: pathlib.Path) -> List[Dict[str, Any]]:
    """
    Read the CSV file and return a list of dictionaries, each representing a video.
    """
    if not csv_path.is_file():
        print(f"Error: CSV file not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    video_data: List[Dict[str, Any]] = []
    required_cols = {'id', 'video', 'url'}

    with csv_path.open(newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        if not required_cols.issubset(reader.fieldnames):
            print(f"Error: CSV must contain the columns: {', '.join(required_cols)}", file=sys.stderr)
            sys.exit(1)

        for row in reader:
            if all(row.get(col, '').strip() for col in required_cols):
                video_data.append(row)
            else:
                print(f"Warning: Skipping row with missing data: {row}", file=sys.stderr)

    return video_data


def download_videos(
    video_data: List[Dict[str, Any]],
    temp_dir: pathlib.Path,
    final_dir: pathlib.Path,
    download: bool
) -> None:
    """
    Download videos, renaming and moving them to the final directory.
    """
    temp_dir.mkdir(parents=True, exist_ok=True)
    final_dir.mkdir(parents=True, exist_ok=True)

    for video in video_data:
        url = video['url']
        csv_id = video['id']
        csv_video_name = video['video']

        # 1. Construct the final filename from CSV data
        safe_video_name = csv_video_name.replace('/', '_').replace('\\', '_')
        final_filename = f"{csv_id} - {safe_video_name}.mp4"
        final_filepath = final_dir / final_filename

        # 2. Check if the file already exists in the final directory
        if final_filepath.is_file():
            print(f"Skipping '{final_filename}' – already exists.")
            continue

        # 3. Download to a temporary file using the final filename
        temp_filepath = temp_dir / final_filename
        
        download_cmd = [
            "yt-dlp",
            "-f", "bestvideo+bestaudio/best",
            "--merge-output-format", "mp4",
            "-o", str(temp_filepath),
            url,
        ]

        try:
            if not download:
                print(f"Would download: {url} -> '{final_filename}'")
                continue
            print(f"Downloading: {url} -> '{final_filename}'")
            subprocess.run(download_cmd, check=True, capture_output=True, text=True)

            # 4. Move the downloaded file from temp to final directory
            if temp_filepath.is_file():
                print(f"Moving '{temp_filepath.name}' to {final_dir}")
                shutil.move(str(temp_filepath), str(final_filepath))
            else:
                print(f"Error: Could not find downloaded file '{final_filename}' in {temp_dir}", file=sys.stderr)

        except subprocess.CalledProcessError as e:
            print(f"--- Error downloading {url} ---\n{e.stderr}\n--- End of Error ---", file=sys.stderr)



def main(
    csv_path: pathlib.Path,
    download: bool,
    temp_dir: pathlib.Path,
    final_dir: pathlib.Path,
) -> None:
    """
    Main entry point.
    """
    video_data = read_video_data(csv_path)

    download_videos(video_data, temp_dir, final_dir, download)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download videos from a CSV, naming them '<video_id> - <title>.mp4'."
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download videos using yt-dlp. Without this flag, only URLs are printed.",
    )
    parser.add_argument(
        "--temp-dir",
        type=pathlib.Path,
        default=".",
        help="Temporary directory for downloads (default: ./temp_downloads).",
    )
    parser.add_argument(
        "--final-dir",
        type=pathlib.Path,
        default="../raw_video",
        help="Final destination for merged videos (default: ../raw_video)."
    )
    parser.add_argument(
        "--csv",
        type=pathlib.Path,
        default=None,
        help="Path to the input CSV file (default: ../set/match.csv).",
    )

    args = parser.parse_args()

    script_dir = pathlib.Path(__file__).parent
    
    csv_file = args.csv if args.csv is not None else script_dir / ".." / "set" / "match.csv"
    
    temp_dir = script_dir / args.temp_dir
    final_dir = args.final_dir

    print(f"CSV file: {csv_file}")
    print(f"Temporary directory: {temp_dir}")
    print(f"Final directory: {final_dir}")

    main(csv_file, args.download, temp_dir, final_dir)