from models.model_deeplab3 import deeplabv3_plus
from utilities.data_loader import output_path, train_dataloader, val_dataloader, test_dataloader
from utilities.hyperparameters import device, N_EPOCHS, NUM_CLASSES, MAX_LR, MODEL_NAME

import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
import segmentation_models_pytorch as smp

from utils import meanIoU
from utils import train_validate_model

criterion = smp.losses.DiceLoss('multiclass', classes=[0, 1, 2], log_loss=True, smooth=1.0)

# create model, optimizer, lr_scheduler and pass to training function
model = deeplabv3_plus(in_channels=3, output_stride=8, num_classes=NUM_CLASSES).to(device)
optimizer = optim.Adam(model.parameters(), lr=MAX_LR)
scheduler = OneCycleLR(optimizer, max_lr=MAX_LR, epochs=N_EPOCHS, steps_per_epoch=len(train_dataloader),
                       pct_start=0.3, div_factor=10, anneal_strategy='cos')

_ = train_validate_model(model, N_EPOCHS, MODEL_NAME, criterion, optimizer,
                         device, train_dataloader, val_dataloader, meanIoU, 'meanIoU',
                         NUM_CLASSES, lr_scheduler=scheduler, output_path=output_path)
