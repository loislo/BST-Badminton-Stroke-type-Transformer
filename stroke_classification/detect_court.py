from courtview import get_frameNum_of_highest_courtView

from pathlib import Path

import subprocess
import concurrent.futures


def pack_court_detection_args_ls(
    process_name: str,
    video_paths: list,
    max_submits: int,
):
    folder = Path('court_detection_temp')
    if not folder.is_dir():
        folder.mkdir()

    txt_ls = [f'{folder.name}\\border{i}.txt' for i in range(max_submits)]
    img_ls = [f'{folder.name}\\img{i}.jpg' for i in range(max_submits)]

    # Repeat `ls` and make len(ls) = len(video_paths)
    txt_ls = (txt_ls * (len(video_paths) // len(txt_ls) + 1))[:len(video_paths)]
    img_ls = (img_ls * (len(video_paths) // len(img_ls) + 1))[:len(video_paths)]

    return [[process_name]+[v]+[txt]+[img] for v, txt, img in zip(video_paths, txt_ls, img_ls)]


def court_detect_with_highest_courtView(process_args: list):
    process_args.append(str(get_frameNum_of_highest_courtView(video_path=process_args[1])))
    return subprocess.run(process_args)


if __name__ == '__main__':
    process_name = 'court_detection.exe'
    max_submits = 3
    video_paths = [
        "C:\\MyResearch\\ShuttleSet\\my_dataset\\train\\Bottom_發長球\\22_1_2_1.mp4",
        "C:\\MyResearch\\ShuttleSet\\my_dataset\\train\\Bottom_發短球\\1_1_2_1.mp4",
        "C:\\MyResearch\\ShuttleSet\\my_dataset\\train\\Bottom_發長球\\22_1_14_1.mp4"
    ]
    output_paths = [f'output{i}.txt' for i in range(max_submits)]
    process_args_ls = pack_court_detection_args_ls(process_name, video_paths, max_submits)

    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(court_detect_with_highest_courtView, args) for args in process_args_ls[:max_submits]]
        iter_p = iter(process_args_ls[max_submits:])
        
        target_file = None
        while target_file is None:
            done, not_done = concurrent.futures.wait(futures, return_when=concurrent.futures.FIRST_COMPLETED)
            for task in done:
                r: subprocess.CompletedProcess = task.result()
                
                if r.returncode == 0:  # successfully detected
                    target_file = r.args[2]
                    break
                
                del futures[futures.index(task)]  # remove failed

                # try to detect another one
                try:
                    args = next(iter_p)
                    futures.append(executor.submit(court_detect_with_highest_courtView, args))
                except StopIteration:
                    pass

        print(target_file)

