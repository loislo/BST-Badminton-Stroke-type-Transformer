import matplotlib.pyplot as plt
from matplotlib.axes import Axes

import pandas as pd


def read_csv_and_get_xy(filename: str):
    df = pd.read_csv(filename)
    return list(df["Step"]), list(df['Value'])


def set_one_ax(ax: Axes, x: dict, y: dict, colors: list[str]):
    ax.plot(
        x['TemPose(original)'], y['TemPose(original)'],
        label='TemPose-TF(original)',
        color=colors[2]
    )
    ax.plot(
        x['TemPose'], y['TemPose'],
        label='TemPose-TF',
        color=colors[1]
    )
    ax.plot(
        x['BST'], y['BST'],
        label='BST-3',
        color=colors[0]
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()


def plot():
    fig = plt.figure(figsize=(8, 4))
    fig.suptitle("Loss Curves")
    ax1, ax2 = fig.subplots(1, 2)
    ax1: Axes; ax2: Axes

    colors = ['darkorange', 'cornflowerblue', 'lightgray']

    bst_train_x, bst_train_y = read_csv_and_get_xy("BST-train.csv")
    tem_train_x, tem_train_y = read_csv_and_get_xy("TemPose-train.csv")
    ori_tem_train_x, ori_tem_train_y = read_csv_and_get_xy("original_TemPose-train.csv")

    x = dict(); y = dict()
    x['BST'] = bst_train_x
    y['BST'] = bst_train_y
    x['TemPose'] = tem_train_x
    y['TemPose'] = tem_train_y
    x['TemPose(original)'] = ori_tem_train_x
    y['TemPose(original)'] = ori_tem_train_y

    ax1.set_title("Train Loss")
    set_one_ax(ax1, x, y, colors)

    bst_val_x, bst_val_y = read_csv_and_get_xy("BST-val.csv")
    tem_val_x, tem_val_y = read_csv_and_get_xy("TemPose-val.csv")
    ori_tem_val_x, ori_tem_val_y = read_csv_and_get_xy("original_TemPose-val.csv")

    x = dict(); y = dict()
    x['BST'] = bst_val_x
    y['BST'] = bst_val_y
    x['TemPose'] = tem_val_x
    y['TemPose'] = tem_val_y
    x['TemPose(original)'] = ori_tem_val_x
    y['TemPose(original)'] = ori_tem_val_y

    ax2.set_title("Validation Loss")
    set_one_ax(ax2, x, y, colors)

    plt.show()


if __name__ == "__main__":
    plot()
