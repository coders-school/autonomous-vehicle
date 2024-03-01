# Import libraries
import numpy as np

# Libraries for Deep Learning
import torch
import torch.nn as nn
import torch.nn.functional as F

# Libraries for loading image, plotting
import cv2
import matplotlib.pyplot as plt

# Library for basic segmentation
import segmentation_models_pytorch as smp


def check_img_by_idx(idx: int, images=np.ndarray, labels=np.array) -> None:
    """
    Function to plot image that contains frame from traffic (car camera POV) and label that
    contains segmented road.
    :param idx: Index of corresponding image and mask.
    :param images: Numpy array that contains images.
    :param labels: Numpy array that contains labels.
    :return:
    """
    nrows = 1
    ncols = 2
    f, (ax0, ax1) = plt.subplots(nrows, ncols)
    ax0.imshow(images[idx])
    ax0.set_title("Image")
    ax1.imshow(labels[idx])
    ax1.set_title("Label")
    plt.show()


if __name__ == "__main__":
    # Load images and labels (binary mask)
    images = np.load("dataset/image_180_320.npy", allow_pickle=True)
    labels = np.load("dataset/label_180_320.npy", allow_pickle=True)

    check_img_by_idx(316, images, labels)