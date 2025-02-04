import os
import xml.etree.ElementTree as ET
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models.segmentation import (
    deeplabv3_mobilenet_v3_large,
    DeepLabV3_MobileNet_V3_Large_Weights,
)
from torchvision.models.segmentation import (
    deeplabv3_resnet50,
    DeepLabV3_ResNet50_Weights,
)
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
from tqdm import tqdm
import logging
from SegmentationDataset import *

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
os.makedirs("models", exist_ok=True)

# Add class mapping
CLASS_MAPPING = {
    "adult": 1,
    "egg masses": 2,
    "instar nymph (1-3)": 3,
    "instar nymph (4)": 4,
}


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    num_epochs,
    device,
    seed,
):
    # Set random seeds for reproducibility
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

    best_val_loss = float("inf")  # Initialize best validation loss to positive infinity
    best_model_path = "models/best_model.pth"  # Best model save path

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 50)
        print(f"Current device: {device}")

        # Training phase
        model.train()
        train_loss = 0.0
        train_steps = 0
        pbar = tqdm(train_loader, desc="Training", leave=True)
        for inputs, masks in pbar:
            inputs = inputs.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)["out"]
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_steps += 1

            # Update progress bar
            pbar.set_postfix(
                {
                    "Loss": f"{loss.item():.4f}",
                    "Avg Loss": f"{train_loss/train_steps:.4f}",
                }
            )

        epoch_train_loss = train_loss / train_steps

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_steps = 0
        pbar = tqdm(val_loader, desc="Validation", leave=True)
        with torch.no_grad():
            for inputs, masks in pbar:
                inputs = inputs.to(device)
                masks = masks.to(device)
                outputs = model(inputs)["out"]
                loss = criterion(outputs, masks)

                val_loss += loss.item()
                val_steps += 1

                # Update progress bar
                pbar.set_postfix(
                    {
                        "Loss": f"{loss.item():.4f}",
                        "Avg Loss": f"{val_loss/val_steps:.4f}",
                    }
                )

        epoch_val_loss = val_loss / val_steps

        # Print epoch summary
        print("\nEpoch Summary:")
        print(f"  Training Loss: {epoch_train_loss:.4f}")
        print(f"  Validation Loss: {epoch_val_loss:.4f}")

        scheduler.step(epoch_val_loss)

        # Save the model for the current epoch
        epoch_model_path = f"models/model_epoch_{epoch+1}.pth"
        torch.save(model.state_dict(), epoch_model_path)
        print(f"Model for epoch {epoch+1} saved to {epoch_model_path}")

        # Check if it's the best validation loss
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model saved to {best_model_path}")
