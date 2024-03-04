from models.model_pspnet import PSPNet
from utilities.data_loader import output_path, train_dataloader, val_dataloader
from utilities.hyperparameters import device, N_EPOCHS, NUM_CLASSES, MAX_LR, MODEL_NAME

import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR

from utils import meanIoU
from utils import pspnet_loss
from utils import train_validate_model

criterion = pspnet_loss(num_classes=NUM_CLASSES, aux_weight=0.4)

# create model, optimizer, lr_scheduler and pass to training function
model = PSPNet(in_channels=3, num_classes=NUM_CLASSES, use_aux=True).to(device)
optimizer = optim.Adam(model.parameters(), lr=MAX_LR)
scheduler = OneCycleLR(optimizer, max_lr=MAX_LR, epochs=N_EPOCHS, steps_per_epoch=len(train_dataloader),
                       pct_start=0.3, div_factor=10, anneal_strategy='cos')

# Run Train/Evaluate Function
_ = train_validate_model(model, N_EPOCHS, MODEL_NAME, criterion, optimizer,
                         device, train_dataloader, val_dataloader, meanIoU, 'meanIoU',
                         NUM_CLASSES, lr_scheduler=scheduler, output_path=output_path)
