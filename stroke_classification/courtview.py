import torch
from torch import nn, Tensor
from torch.utils.data import IterableDataset, DataLoader
import torch.nn.functional as F
from torchvision.transforms import v2
from torchvision.models import resnet18

import moviepy.editor as mpe
import numpy as np
from pathlib import Path
import cv2

from tqdm import tqdm
from typing import Union


def frameNum_2_time(frame_number: int, fps: Union[int, float]) -> str:
    """
    Converts a frame number to time string (HH:MM:SS.ssssss) based on FPS.

    Parameters
    ----------
    `frame_number`
        The frame number (started from 0).
    `fps`
        The frames per second of the video.

    Returns
    -------
    The time string corresponding to the frame number.
    """
    total_seconds = frame_number / fps

    hours = int(total_seconds // 3600)
    minutes = int(total_seconds % 3600 // 60)
    seconds = total_seconds % 60 + 0.5 / fps

    return f"{hours:02d}:{minutes:02d}:{seconds:09.6f}"


class CourtViewDataset_infer(IterableDataset):
    def __init__(self, video_path) -> None:
        super().__init__()
        self.video: mpe.VideoClip = mpe.VideoFileClip(video_path).resize((64, 64))
        self.iterator = self.video.iter_frames()

        self.tfm = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __iter__(self):
        return self
    
    def __next__(self):
        try:
            frame = next(self.iterator)
            return self.tfm(frame)
        except StopIteration:
            raise StopIteration()

    def __len__(self):
        return int(self.video.duration * self.video.fps)


class CourtViewClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.resnet18 = resnet18(num_classes=1)

    def forward(self, x: Tensor):
        x = self.resnet18(x)
        return x.squeeze(-1)


@torch.no_grad()
def infer(
    model: CourtViewClassifier,
    loader: DataLoader,
    device: torch.device
):
    model.eval()
    predictions = []
    # pbar = tqdm(range(len(loader)), desc='Processing', unit='batch')
    for batch in loader:
        batch: Tensor = batch.to(device)
        pred: Tensor = F.sigmoid(model(batch))
        predictions.append(pred.squeeze(-1).cpu().numpy())
    #     pbar.update()
    # pbar.close()
    return np.concatenate(predictions)


def get_frameNum_of_highest_courtView(video_path: str, model=None):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    data = CourtViewDataset_infer(video_path)
    loader = DataLoader(data, batch_size=512, pin_memory=torch.cuda.is_available())

    if model is None:
        weight_path = Path('weight')/f'courtview_weight.pt'
        model = CourtViewClassifier().to(device)
        model.load_state_dict(torch.load(weight_path))

    predictions = infer(model, loader, device)
    return predictions.argmax()


def get_CourtViewClassifier_pretrained():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    weight_path = Path('weight')/f'courtview_weight.pt'
    model = CourtViewClassifier().to(device)
    model.load_state_dict(torch.load(weight_path))
    return model


if __name__ == "__main__":
    video_path = "C:\\MyResearch\\ShuttleSet\\my_dataset\\train\\Bottom_發長球\\22_1_2_1.mp4"
    frame_num = get_frameNum_of_highest_courtView(video_path)
    
    video = mpe.VideoFileClip(video_path)
    t_str = frameNum_2_time(frame_num, fps=video.fps)
    frame = video.get_frame(t_str)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.imwrite(f'{frame_num}.jpg', frame)
