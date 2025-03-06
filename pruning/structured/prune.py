import argparse
import logging
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch_pruning as tp
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models.segmentation import (
    DeepLabV3_MobileNet_V3_Large_Weights,
    DeepLabV3_ResNet50_Weights,
    FCN_ResNet50_Weights,
    deeplabv3_mobilenet_v3_large,
    deeplabv3_resnet50,
    fcn_resnet50,
)
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PruningConfig:
    """Configuration class for model pruning parameters"""

    checkpoint_path: str
    output_path: str
    example_input_size: Tuple[int, ...] = (1, 3, 512, 512)
    pruning_ratio: float = 0.9  # Ratio of parameters to be pruned
    device: str = "cuda"  # Device to perform pruning and training
    fine_tune_epochs: int = 1  # Epochs for intermediate fine-tuning
    final_fine_tune_epochs: int = 1  # Epochs for final fine-tuning
    data_root: str = "../../data"  # Root directory of dataset
    fine_tune_lr: float = 1e-4  # Learning rate for fine-tuning
    batch_size: int = 4
    seed: int = 42  # Random seed for reproducibility
    num_classes: int = 5  # Number of segmentation classes
    early_stop_patience: int = 10  # Patience for early stopping
    iterative_steps: int = 5  # Number of progressive pruning steps
    model_type: str = ""  # Type of model to prune
    input_size: Tuple[int, int] = (520, 520)  # Default input size for images


class SegmentationDatasetConfig:
    """Static configuration for segmentation dataset"""

    # Mapping from class names to label indices
    CLASS_MAPPING = {
        "adult": 1,
        "egg masses": 2,
        "instar nymph (1-3)": 3,
        "instar nymph (4)": 4,
    }

    # Get proper input size based on model type
    @staticmethod
    def get_transform_for_model(model_type: str):
        # Choose input size based on model requirements
        if model_type in ["FCN_EfficientNet", "UNET_MobileNetV3", "UNET_ResNet"]:
            # These models require dimensions divisible by 32
            resize_size = (512, 512)
        else:
            # Default size for other models
            resize_size = (520, 520)

        return transforms.Compose(
            [
                transforms.Resize(resize_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )


class SegmentationDataset(Dataset):
    """Dataset class for segmentation tasks"""

    def __init__(
        self, root_dir: str, split: str = "train", transform=None, model_type: str = ""
    ):
        self.root_dir = os.path.join(root_dir, split)
        # Use model-specific transform if none is provided
        if transform is None:
            transform = SegmentationDatasetConfig.get_transform_for_model(model_type)
        self.transform = transform
        self.model_type = model_type
        self.images, self.annotations = self._load_dataset()
        logger.info(f"Found {len(self.images)} images in {split} set")

    def _load_dataset(self):
        images, annotations = [], []
        for filename in os.listdir(self.root_dir):
            if filename.endswith((".jpg", ".jpeg", ".png")):
                xml_file = os.path.splitext(filename)[0] + ".xml"
                if os.path.exists(os.path.join(self.root_dir, xml_file)):
                    images.append(filename)
                    annotations.append(xml_file)
        return images, annotations

    def parse_xml(self, xml_path: str) -> Image.Image:
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            size = root.find("size")
            if size is None:
                raise ValueError("Missing size in XML")

            width = int(size.find("width").text)
            height = int(size.find("height").text)
            mask = np.zeros((height, width), dtype=np.uint8)

            for obj in root.findall("object"):
                class_name = self._get_class_name(obj)
                if (
                    not class_name
                    or class_name not in SegmentationDatasetConfig.CLASS_MAPPING
                ):
                    continue

                bbox = obj.find("bndbox")
                if bbox is None:
                    continue

                xmin, ymin, xmax, ymax = self._get_bbox_coordinates(bbox, width, height)
                if xmin >= xmax or ymin >= ymax:
                    continue

                mask[ymin:ymax, xmin:xmax] = SegmentationDatasetConfig.CLASS_MAPPING[
                    class_name
                ]

            return Image.fromarray(mask)
        except Exception as e:
            logger.error(f"Error parsing XML file {xml_path}: {e}")
            return Image.fromarray(np.zeros((height, width), dtype=np.uint8))

    @staticmethod
    def _get_class_name(obj) -> Optional[str]:
        name_tag = obj.find("name")
        if name_tag is not None:
            return name_tag.text
        n_tag = obj.find("n")
        return n_tag.text if n_tag is not None else None

    @staticmethod
    def _get_bbox_coordinates(
        bbox, width: int, height: int
    ) -> Tuple[int, int, int, int]:
        return (
            max(0, int(bbox.find("xmin").text)),
            max(0, int(bbox.find("ymin").text)),
            min(width, int(bbox.find("xmax").text)),
            min(height, int(bbox.find("ymax").text)),
        )

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path = os.path.join(self.root_dir, self.images[idx])
        annotation_path = os.path.join(self.root_dir, self.annotations[idx])

        image = Image.open(img_path).convert("RGB")
        mask = self.parse_xml(annotation_path)

        if self.transform:
            image = self.transform(image)
            # Resize mask to match the model-specific input size
            if self.model_type in [
                "FCN_EfficientNet",
                "UNET_MobileNetV3",
                "UNET_ResNet",
            ]:
                resize_size = (512, 512)
            else:
                resize_size = (520, 520)

            mask = transforms.Resize(resize_size)(mask)
            mask = torch.from_numpy(np.array(mask)).long()

        return image, mask


class ModelHandler:
    """Handles model initialization and pruning setup"""

    @staticmethod
    def get_model_type_from_path(checkpoint_path: str) -> str:
        """Determine model type from checkpoint filename"""
        base_name = os.path.basename(checkpoint_path)
        if "FCN_EfficientNet" in base_name:
            return "FCN_EfficientNet"
        elif "FCN_resnet" in base_name:
            return "FCN_ResNet"
        elif "UNET_mobilenet" in base_name:
            return "UNET_MobileNetV3"
        elif "UNET_resnet" in base_name:
            return "UNET_ResNet"
        elif "Deeplabv3_mobilenet" in base_name:
            return "DeepLabV3_MobileNet"
        elif "Deeplabv3_resnet" in base_name:
            return "DeepLabV3_ResNet"
        else:
            # If not a recognized model type, default to DeepLabV3_ResNet
            logger.warning(
                f"Unknown model type in checkpoint: {base_name}, using default DeepLabV3_ResNet"
            )
            return "DeepLabV3_ResNet"

    @staticmethod
    def get_model(num_classes: int, checkpoint_path: str) -> nn.Module:
        """Initialize model with pretrained weights and custom head based on model type"""
        model_type = ModelHandler.get_model_type_from_path(checkpoint_path)
        logger.info(f"Detected model type: {model_type}")

        if model_type == "DeepLabV3_ResNet":
            model = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT)
            model.classifier[-1] = nn.Conv2d(256, num_classes, kernel_size=1)
        elif model_type == "DeepLabV3_MobileNet":
            model = deeplabv3_mobilenet_v3_large(
                weights=DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT
            )
            model.classifier[-1] = nn.Conv2d(256, num_classes, kernel_size=1)
        elif model_type == "FCN_EfficientNet":
            # Initialize FCN with EfficientNet-B0 backbone
            model = smp.FPN(
                "efficientnet-b0",
                encoder_weights="imagenet",
                classes=num_classes,
                activation=None,
            )
            model.segmentation_head = nn.Sequential(
                nn.Conv2d(
                    128, num_classes, kernel_size=1, stride=1
                ),  # EfficientNet-B0 outputs 128 feature maps
                nn.Upsample(
                    scale_factor=4, mode="bilinear", align_corners=True
                ),  # Adjust resolution
            )
        elif model_type == "FCN_ResNet":
            model = fcn_resnet50(weights=FCN_ResNet50_Weights.DEFAULT)
            # FCN ResNet50 model has a different classifier structure than DeepLabV3
            model.classifier[-1] = nn.Conv2d(512, num_classes, kernel_size=1, stride=1)
        elif model_type == "UNET_MobileNetV3":
            # Initialize UNet with MobileNetV3-Small backbone
            model = smp.Unet(
                encoder_name="timm-mobilenetv3_small_100",
                encoder_weights="imagenet",
                in_channels=3,
                classes=num_classes,
                activation=None,
            )
        elif model_type == "UNET_ResNet":
            # Initialize UNet with ResNet34 backbone
            model = smp.Unet(
                encoder_name="resnet34",
                encoder_weights="imagenet",
                in_channels=3,
                classes=num_classes,
                activation=None,
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        # Load checkpoint
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)
        return model

    @staticmethod
    def get_output_layer(model: nn.Module, model_type: str):
        """Get output layer for different model types to ignore during pruning"""
        if model_type in ["DeepLabV3_ResNet", "DeepLabV3_MobileNet", "FCN_ResNet"]:
            return [model.classifier[-1]]  # Output layer for DeepLabV3 models
        elif model_type == "FCN_EfficientNet":
            return [
                model.segmentation_head[0]
            ]  # First layer in the segmentation head for FPN
        elif model_type == "UNET_MobileNetV3":
            return [model.segmentation_head]  # The segmentation head for UNet
        elif model_type == "UNET_ResNet":
            return [
                model.segmentation_head
            ]  # The segmentation head for UNet with ResNet
        else:
            # Default to an empty list if model type is unknown
            logger.warning(f"Unknown model type for output layer: {model_type}")
            return []

    @staticmethod
    def setup_pruner(
        model: nn.Module, config: PruningConfig
    ) -> tp.pruner.MagnitudePruner:
        """Configure magnitude-based pruner with global pruning strategy"""
        model_type = ModelHandler.get_model_type_from_path(config.checkpoint_path)

        # Adjust example input size based on model type
        example_input_size = list(config.example_input_size)
        if model_type in ["FCN_EfficientNet", "UNET_MobileNetV3", "UNET_ResNet"]:
            # Use 512x512 for models requiring input size divisible by 32
            example_input_size[2] = 512
            example_input_size[3] = 512

        example_inputs = torch.randn(*example_input_size).to(config.device)
        imp = tp.importance.MagnitudeImportance(
            p=2
        )  # L2 norm for importance calculation

        # Get model type and identify the output layer
        ignored_layers = ModelHandler.get_output_layer(model, model_type)

        return tp.pruner.MagnitudePruner(
            model=model,
            example_inputs=example_inputs,
            importance=imp,
            global_pruning=True,
            pruning_ratio=config.pruning_ratio,
            iterative_steps=config.iterative_steps,
            ignored_layers=ignored_layers,
        )


class Trainer:
    """Handles model training and validation during pruning process"""

    def __init__(self, config: PruningConfig):
        self.config = config
        self.model_type = ModelHandler.get_model_type_from_path(config.checkpoint_path)
        self._setup_seeds()

    def _setup_seeds(self):
        torch.manual_seed(self.config.seed)
        torch.cuda.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def _get_dataloaders(self) -> Tuple[DataLoader, DataLoader]:
        train_dataset = SegmentationDataset(
            self.config.data_root, split="train", model_type=self.model_type
        )
        val_dataset = SegmentationDataset(
            self.config.data_root, split="valid", model_type=self.model_type
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            drop_last=True,
        )
        return train_loader, val_loader

    def fine_tune(self, model: nn.Module, num_epochs: int) -> nn.Module:
        """Fine-tune model with early stopping and learning rate scheduling"""
        model.train()
        train_loader, val_loader = self._get_dataloaders()
        criterion = self._setup_criterion()
        optimizer = torch.optim.Adam(
            model.parameters(), lr=self.config.fine_tune_lr, weight_decay=1e-4
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.6, patience=5, verbose=True
        )

        best_val_loss = float("inf")
        early_stop_counter = 0
        best_model = None  # Store the entire model for FCN_EfficientNet
        best_state_dict = None  # Store state dict for other models

        # Create temporary directory for saving models during training
        os.makedirs("temp_models", exist_ok=True)
        temp_model_path = os.path.join("temp_models", "temp_model.pth")

        for epoch in range(num_epochs):
            train_loss = self._train_epoch(model, train_loader, criterion, optimizer)
            val_loss = self._validate_epoch(model, val_loader, criterion)

            scheduler.step(val_loss)

            # Early stopping logic
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                # Handle different model types
                if self.model_type in [
                    "FCN_EfficientNet",
                    "UNET_MobileNetV3",
                    "UNET_ResNet",
                ]:
                    # For models not compatible with tp.state_dict, save the entire model
                    best_model = model
                    # Also save to disk as backup
                    torch.save(model, temp_model_path)
                    logger.info(
                        f"Saved best model at epoch {epoch} (loss: {val_loss:.4f})"
                    )
                else:
                    # For other models, use torch_pruning state dict
                    try:
                        best_state_dict = tp.state_dict(model)
                    except Exception as e:
                        logger.warning(f"Error using tp.state_dict: {e}")
                        # Fall back to saving the entire model
                        best_model = model
                        torch.save(model, temp_model_path)
                        logger.info(
                            f"Saved best model at epoch {epoch} (loss: {val_loss:.4f})"
                        )

                early_stop_counter = 0
            else:
                early_stop_counter += 1
                if early_stop_counter >= self.config.early_stop_patience:
                    break

        # Restore best model
        if (
            self.model_type in ["FCN_EfficientNet", "UNET_MobileNetV3", "UNET_ResNet"]
            or best_model is not None
        ):
            # Use the saved best model directly if we have it in memory
            if best_model is not None:
                model = best_model
            # Otherwise load from disk
            elif os.path.exists(temp_model_path):
                logger.info("Loading best model from disk")
                model = torch.load(temp_model_path)
        elif best_state_dict is not None:
            # For standard models, use torch_pruning to load state dict
            try:
                tp.load_state_dict(model, state_dict=best_state_dict)
            except Exception as e:
                logger.warning(f"Error using tp.load_state_dict: {e}")
                # If tp.load_state_dict fails, load from disk if available
                if os.path.exists(temp_model_path):
                    logger.info(
                        "Loading best model from disk due to tp.load_state_dict failure"
                    )
                    model = torch.load(temp_model_path)

        # Clean up temporary files
        if os.path.exists(temp_model_path):
            os.remove(temp_model_path)

        return model

    def _setup_criterion(self) -> nn.Module:
        class_weights = torch.tensor([0.5, 1.0, 1.1, 1.5, 2.0], dtype=torch.float)
        return nn.CrossEntropyLoss(weight=class_weights.to(self.config.device))

    def _train_epoch(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
    ) -> float:
        model.train()
        total_loss = 0.0
        steps = 0

        for images, masks in tqdm(train_loader, desc="Training", leave=False):
            images = images.to(self.config.device)
            masks = masks.to(self.config.device)

            optimizer.zero_grad()

            # Handle different model output formats
            if self.model_type in [
                "DeepLabV3_ResNet",
                "DeepLabV3_MobileNet",
                "FCN_ResNet",
            ]:
                outputs = model(images)["out"]
            else:  # FCN_EfficientNet, UNET_MobileNetV3 and other models
                outputs = model(images)

            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            steps += 1

        return total_loss / steps if steps else 0.0

    def _validate_epoch(
        self, model: nn.Module, val_loader: DataLoader, criterion: nn.Module
    ) -> float:
        model.eval()
        total_loss = 0.0
        steps = 0

        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc="Validation", leave=False):
                images = images.to(self.config.device)
                masks = masks.to(self.config.device)

                # Handle different model output formats
                if self.model_type in [
                    "DeepLabV3_ResNet",
                    "DeepLabV3_MobileNet",
                    "FCN_ResNet",
                ]:
                    outputs = model(images)["out"]
                else:  # FCN_EfficientNet, UNET_MobileNetV3 and other models
                    outputs = model(images)

                loss = criterion(outputs, masks)
                total_loss += loss.item()
                steps += 1

        return total_loss / steps if steps else 0.0


def prune_model(config: PruningConfig) -> nn.Module:
    """Main pruning pipeline implementing progressive pruning with fine-tuning"""
    model_handler = ModelHandler()
    model = model_handler.get_model(config.num_classes, config.checkpoint_path)
    model.to(config.device)

    # Get model type for special handling
    model_type = ModelHandler.get_model_type_from_path(config.checkpoint_path)

    pruner = model_handler.setup_pruner(model, config)
    trainer = Trainer(config)
    # Set model type in trainer for proper handling
    trainer.model_type = model_type

    # Create directory for intermediate models
    intermediate_model_dir = os.path.join(
        os.path.dirname(config.output_path), "intermediate_models"
    )
    os.makedirs(intermediate_model_dir, exist_ok=True)

    # Progressive pruning with intermediate fine-tuning
    for i in range(config.iterative_steps):
        pruner.step()

        # Adjust example input size based on model type
        example_input_size = list(config.example_input_size)
        if model_type in ["FCN_EfficientNet", "UNET_MobileNetV3", "UNET_ResNet"]:
            # Use 512x512 for models requiring input size divisible by 32
            example_input_size[2] = 512
            example_input_size[3] = 512

        macs, nparams = tp.utils.count_ops_and_params(
            model, torch.randn(*example_input_size).to(config.device)
        )
        print(f"Round {i + 1}/{config.iterative_steps}, Params: {nparams / 1e6:.2f} M")

        # Fine-tune with special handling for FCN_EfficientNet
        model = trainer.fine_tune(model, config.fine_tune_epochs)

        # Save intermediate model
        if model_type in ["FCN_EfficientNet", "UNET_MobileNetV3", "UNET_ResNet"]:
            # For models not compatible with tp.state_dict, save the full model
            intermediate_path = os.path.join(
                intermediate_model_dir,
                f"intermediate_{i + 1}_of_{config.iterative_steps}.model.pth",
            )
            torch.save(model, intermediate_path)
        else:
            # For other models, use torch_pruning state dict
            intermediate_path = os.path.join(
                intermediate_model_dir,
                f"intermediate_{i + 1}_of_{config.iterative_steps}.pth",
            )
            try:
                torch.save(tp.state_dict(model), intermediate_path)
            except Exception as e:
                logger.warning(f"Error using tp.state_dict for saving: {e}")
                # Fall back to saving the entire model
                torch.save(model, intermediate_path.replace(".pth", ".model.pth"))

    # Final fine-tuning after all pruning steps
    model = trainer.fine_tune(model, config.final_fine_tune_epochs)

    # Save final pruned model
    model.cpu()

    # Save differently based on model type
    if model_type in ["FCN_EfficientNet", "UNET_MobileNetV3", "UNET_ResNet"]:
        # For models not compatible with tp.state_dict, save full model
        model_path = config.output_path.replace(".pth", ".model.pth")
        torch.save(model, model_path)
        logger.info(f"Saved full pruned model to {model_path}")
    else:
        # For other models, try to use torch_pruning state dict
        try:
            torch.save(tp.state_dict(model), config.output_path)
            logger.info(f"Saved pruned model state dict to {config.output_path}")
        except Exception as e:
            logger.warning(f"Error using tp.state_dict for saving: {e}")
            # Fall back to saving the entire model
            model_path = config.output_path.replace(".pth", ".model.pth")
            torch.save(model, model_path)
            logger.info(f"Saved full pruned model to {model_path}")

    return model


def get_pruned_model_name(checkpoint_path: str, pruning_ratio: float) -> str:
    """Generate standardized name for pruned model checkpoint"""
    base_name = os.path.splitext(os.path.basename(checkpoint_path))[0]
    return f"pruned_{base_name}_magnitude_pruner_{pruning_ratio:.2f}.pth"


def parse_args():
    parser = argparse.ArgumentParser(description="Model pruning script")
    parser.add_argument(
        "--pruning_ratio", type=float, default=0.5, help="Pruning ratio (default: 0.5)"
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="../unpruned_models/Deeplabv3_resnet_epoch_36.pth",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="../pruned_models",
        help="Directory to save pruned model",
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device to use (default: cuda)"
    )
    parser.add_argument(
        "--data_root", type=str, default="../../data", help="Root directory for dataset"
    )
    parser.add_argument(
        "--batch_size", type=int, default=4, help="Batch size (default: 4)"
    )
    parser.add_argument(
        "--fine_tune_epochs",
        type=int,
        default=20,
        help="Number of fine-tuning epochs per pruning step",
    )
    parser.add_argument(
        "--final_fine_tune_epochs",
        type=int,
        default=50,
        help="Number of final fine-tuning epochs",
    )
    parser.add_argument(
        "--iterative_steps", type=int, default=5, help="Number of pruning iterations"
    )
    args = parser.parse_args()

    args.output_path = os.path.join(
        args.output_dir, get_pruned_model_name(args.checkpoint_path, args.pruning_ratio)
    )

    os.makedirs(args.output_dir, exist_ok=True)

    return args


if __name__ == "__main__":
    args = parse_args()
    config = PruningConfig(
        checkpoint_path=args.checkpoint_path,
        output_path=args.output_path,
        pruning_ratio=args.pruning_ratio,
        device=args.device,
        data_root=args.data_root,
        batch_size=args.batch_size,
        fine_tune_epochs=args.fine_tune_epochs,
        final_fine_tune_epochs=args.final_fine_tune_epochs,
        iterative_steps=args.iterative_steps,
    )
    final_model = prune_model(config)
