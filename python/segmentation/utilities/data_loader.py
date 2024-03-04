import numpy as np
import matplotlib.pyplot as plt

# utility functions to get BDD100k Pytorch dataset and dataloaders
from utils import get_datasets, get_dataloaders

from collections import namedtuple

from utils import inverse_transform

# TODO: convert to function
output_path = 'dataset'

images = np.load("dataset/image_180_320.npy")
labels = np.load("dataset/label_180_320.npy")

train_set, val_set, test_set = get_datasets(images, labels)
sample_image, sample_label = train_set[0]
print(f"There are {len(train_set)} train images, {len(val_set)} validation images, {len(test_set)} test Images")
print(f"Input shape = {sample_image.shape}, output label shape = {sample_label.shape}")

train_dataloader, val_dataloader, test_dataloader = get_dataloaders(train_set, val_set, test_set)

# Each label is a tuple with name, class id and color
Label = namedtuple("Label", ["name", "train_id", "color"])
drivables = [
    Label("direct", 0, (32, 146, 190)),  # red
    Label("alternative", 1, (119, 231, 124)),  # cyan
    Label("background", 2, (0, 0, 0)),  # black
]

train_id_to_color = [c.color for c in drivables if (c.train_id != -1 and c.train_id != 255)]
train_id_to_color = np.array(train_id_to_color)

rgb_image, label = train_set[np.random.choice(len(train_set))]
rgb_image = inverse_transform(rgb_image).permute(1, 2, 0).cpu().detach().numpy()
label = label.cpu().detach().numpy()

# # plot sample image
# fig, axes = plt.subplots(1,2, figsize=(20,10))
# axes[0].imshow(rgb_image)
# axes[0].set_title("Image")
# axes[0].axis('off')
# axes[1].imshow(train_id_to_color[label])
# axes[1].set_title("Label")
# axes[1].axis('off')
# plt.show()
