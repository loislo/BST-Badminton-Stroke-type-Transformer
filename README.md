# BST: Badminton Stroke-type Transformer for Skeleton-based Action Recognition in Racket Sports
[![Static Badge](https://img.shields.io/badge/arXiv-2502.21085-gray?labelColor=%23B31B1B)](https://arxiv.org/abs/2502.21085)

[![Static Badge](https://img.shields.io/badge/Python-3.11-gray?labelColor=%234584B6)](https://www.python.org/downloads/) [![Static Badge](https://img.shields.io/badge/PyTorch-2.4.0%2Bcu121-gray?labelColor=%23EE4C2C)](https://pytorch.org/)

## Training Speed
- On a NVIDIA RTX 4070 Ti Super setup, training **BST-3** takes 29 sec per epoch, while training **TemPose-TF** takes 31 sec per epoch.
- The following figure shows training **BST-3** converges faster than **TemPose-TF**:
![Loss Curves](stroke_classification/tensorboard2pyplot/loss_curves.png)

## Reproducing the Results
### Generating Stroke Clips
(In `ShuttleSet` folder)
1. Download the videos from links in `set/match.csv` and put them into `raw_video` folder.
    - `flaw_shot_records.csv` shows some errors in the original [ShuttleSet](https://github.com/wywyWang/CoachAI-Projects/tree/main/ShuttleSet).
    - Append each video ID to the front of its name, such as `<ID> - <name>.mp4`.
2. Modify `my_raw_video_resolution.csv` to the resolutions you downloaded.
3. Run `gen_my_dataset.py` 6 (2 players x 3 splitted sets) times, and please check the following settings in the code each time:
    - `out_root_dir`: output root directory.
    - `player`: Top / Bottom.
    - `set_name`: train / val / test.
    - `type_ls`: each element should be the same class name as in ShuttleSet.
    - `strategy`: please check if it matches the `out_root_dir` or not.
4. Move all clips in `Top_小平球` or `Bottom_小平球` to `未知球種`, which means "None" type, and delete these empty folders.

Thus, we have stroke clips now.

### Data Preprocessing
(In `stroke_classification` folder)
1. Make sure your shuttlecock tracking model works well, you can choose the one you like:
    - [TrackNetV3 (using attension)](https://github.com/alenzenx/TrackNetV3) $\Leftarrow$ used in this paper.
    - [TrackNetV3 (with rectification module)](https://github.com/qaz812345/TrackNetV3).
2. Make sure your [MMPose](https://github.com/open-mmlab/mmpose/tree/main) works well.
    - Set your `venv_prepare_train` virtual environment first for running Human Pose Estimation (HPE) successfully.
    - See `prepare_train_env.txt` as an example based on my setup.
3. Run 1-3 steps in `prepare_train.py` individually.
    - Check data paths first.
    - For step 1: check the parameters are being passed into your shuttlecock tracking model.
    - For step 2: I don't recommend using 3D poses because there are still some bugs using MMPose 3D pose API.
    - For step 3: set `seq_len` to `30` if using fixed-width clipping strategy, otherwise `100`.

Thus, we have collated .npy files now.

### Training Classification Models
(In `stroke_classification` folder)
- Run each `<architecture_name>_main.py` for training, validating, testing each model, and please check hyper-parameters:
    - `hyp.seq_len`: should be the same as in your dataset.
    - `train_partial`: `1` for the whole training set.
    - `model_name`
        - `ShuttlePose_3_3`: **BST-0** in this paper.
        - `BST_6_2`: **BST-1** in this paper.
        - `BST_7`: **BST-2** in this paper.
        - `BST_8`: **BST-3** in this paper.
    - `in_channels`: `2` for 2D, `3` for 3D.
    - `model_info`: to distinguish your model weights.
    - `serial_no`: to distinguish your model weights when trying different initial weights.

## Citation
```
@misc{chang2025bstbadmintonstroketypetransformer,
      title={BST: Badminton Stroke-type Transformer for Skeleton-based Action Recognition in Racket Sports}, 
      author={Jing-Yuan Chang},
      year={2025},
      eprint={2502.21085},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2502.21085}, 
}
```

## References
- Shuttlecock Trajectory
    - [TrackNetV3 (using attention)](https://github.com/alenzenx/TrackNetV3)
    - [TrackNetV3 (with rectification module)](https://github.com/qaz812345/TrackNetV3)
- Human Pose Estimation
    - [MMPose](https://github.com/open-mmlab/mmpose/tree/main)
- Classification Models
    - [ST-GCN](https://github.com/yysijie/st-gcn)
    - [BlockGCN](https://github.com/ZhouYuxuanYX/BlockGCN)
    - [SkateFormer](https://github.com/KAIST-VICLab/SkateFormer)
    - [ProtoGCN](https://github.com/firework8/ProtoGCN)
    - [TemPose](https://github.com/MagnusPetersenIbh/TemPose-BadmintonActionRecognition)
- Dataset
    - [ShuttleSet](https://github.com/wywyWang/CoachAI-Projects/tree/main/ShuttleSet)
- Court Detection (if the dataset does not contain court information)
    - [MonoTrack](https://github.com/jhwang7628/monotrack)
