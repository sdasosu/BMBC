import os
import xml.etree.ElementTree as ET
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large, DeepLabV3_MobileNet_V3_Large_Weights
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
from tqdm import tqdm
import logging
from SegmentationDataset import *
from train import *

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
os.makedirs("models", exist_ok=True)

#------------------------------------------
results = []


#------------------------------------------
# Add class mapping
#------------------------------------------
CLASS_MAPPING = {
    'adult': 1,
    'egg masses': 2,
    'instar nymph (1-3)': 3,
    'instar nymph (4)': 4
}


#------------------ Training /Finetuning Parameters -----------------
device = torch.device('mps' if torch.backends.mps.is_available() else
                         'cuda' if torch.cuda.is_available() else 'cpu')


# Create dataset and data loader
train_dataset = SegmentationDataset('../../training/data', split='train')
test_dataset= SegmentationDataset('../../training/data', split='test')
val_dataset = SegmentationDataset('../../training/data', split='valid')

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, drop_last=True)
test_loader  = DataLoader(test_dataset, batch_size= 4, shuffle=False, drop_last=True)
val_loader   = DataLoader(val_dataset , batch_size=4, shuffle=False, drop_last=True)




#---------------------------------------------------------------
# Bring the Model --> Prune it return it
#---------------------------------------------------------------
model = deeplabv3_resnet50(weights=None)  # No pre-trained weights
model.classifier[-1] = nn.Conv2d(256, len(CLASS_MAPPING) + 1, kernel_size=(1, 1), stride=(1, 1))
model = model.to(device)

# Load the saved model from unpruned_model folder
checkpoint_path = "unpruned_model/model_epoch_24.pth"
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint)








#---------------------------------------------------------------
# Define weighted loss
class_weights = torch.tensor([0.5, 1.0, 1.1, 1.5, 2.0], dtype=torch.float).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.6, patience=5, verbose=True)
#---------------------------------------------------------------


train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=1, device=device, seed=42)


# ------------------- FineTuning Sub routine ---------------------
#mean_iou_per_class, per_class_acc = evaluate_model( model, test_loader, device, save_dir=f'results/best_model')

#for i, class_name in enumerate(class_names):
#                result[f'iou_{class_name}'] = mean_iou_per_class[i]
#                result[f'acc_{class_name}'] = per_class_acc[i]

#train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=1, device=device, seed=42)
