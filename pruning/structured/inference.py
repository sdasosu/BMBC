from typing import Any, Dict, List, Optional, Sized, Tuple, cast

import matplotlib
import numpy as np
import torch
import torch.nn as nn
import torch_pruning as tp
from prune import SegmentationDataset, SegmentationDatasetConfig
from torch.utils.data import DataLoader
from torchvision import transforms

matplotlib.use("Agg")  # Use non-interactive backend
import os
import re
from dataclasses import dataclass

import matplotlib.pyplot as plt
import pandas as pd
import segmentation_models_pytorch as smp
from torchvision.models.segmentation import (
    deeplabv3_mobilenet_v3_large,
    deeplabv3_resnet50,
    fcn_resnet50,
)
from tqdm import tqdm

# Color mapping for visualization
COLOR_MAP = {
    0: [0, 0, 0],  # Background - Black
    1: [255, 0, 0],  # Adult - Red
    2: [0, 255, 0],  # Egg masses - Green
    3: [0, 0, 255],  # Instar nymph (1-3) - Blue
    4: [255, 255, 0],  # Instar nymph (4) - Yellow
}

# Import class mapping from config
CLASS_MAPPING = SegmentationDatasetConfig.CLASS_MAPPING


@dataclass
class EvaluationConfig:
    """Configuration for model evaluation"""

    model_dir: str = "../pruned_models"
    data_root: str = "../../data"
    save_dir: str = "results"
    batch_size: int = 4
    max_vis_images: int = 10
    num_workers: int = 4


class Visualizer:
    """Handles all visualization related operations"""

    @staticmethod
    def mask_to_color(mask: np.ndarray) -> np.ndarray:
        """Convert class index mask to RGB color mask"""
        color_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
        for class_idx, color in COLOR_MAP.items():
            color_mask[mask == class_idx] = color
        return color_mask

    @staticmethod
    def save_prediction(
        image: torch.Tensor,
        mask: torch.Tensor,
        prediction: torch.Tensor,
        save_path: str,
    ) -> None:
        """Save side-by-side visualization of original, ground truth and prediction"""
        plt.figure(figsize=(15, 5))

        # Plot original image
        plt.subplot(1, 3, 1)
        img = image.cpu().permute(1, 2, 0).numpy()
        img = (img * [0.229, 0.224, 0.225]) + [0.485, 0.456, 0.406]
        img = np.clip(img * 255, 0, 255).astype(np.uint8)
        plt.imshow(img)
        plt.title("Original Image")
        plt.axis("off")

        # Plot ground truth
        plt.subplot(1, 3, 2)
        mask_colored = Visualizer.mask_to_color(mask.cpu().numpy())
        plt.imshow(mask_colored)
        plt.title("Ground Truth")
        plt.axis("off")

        # Plot prediction
        plt.subplot(1, 3, 3)
        pred_colored = Visualizer.mask_to_color(prediction.cpu().numpy())
        plt.imshow(pred_colored)
        plt.title("Prediction")
        plt.axis("off")

        # Save and cleanup
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0.1)
        plt.close()


class ModelEvaluator:
    """Handles model evaluation and metrics calculation"""

    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.device = self._setup_device()
        self.visualizer = Visualizer()
        self.results_data: List[Dict[str, Any]] = []  # Store results for all models

    @staticmethod
    def _setup_device() -> torch.device:
        """Setup the most suitable device for evaluation"""
        return torch.device(
            "mps"
            if torch.backends.mps.is_available()
            else "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    def _get_test_loader(self) -> DataLoader:
        """Initialize test data loader"""
        # Get all pruned models to check if we have special model types
        pruned_models = self._get_sorted_pruned_models()
        has_fpn_efficientnet = any(
            "FPN_EfficientNet" in model[0] for model in pruned_models
        )
        has_fcn_resnet = any("FCN_resnet" in model[0] for model in pruned_models)
        has_unet_mobilenet = any(
            "UNET_mobilenet" in model[0] for model in pruned_models
        )
        has_unet_resnet = any("UNET_resnet" in model[0] for model in pruned_models)

        # Choose model type for dataset based on which models we have
        model_type = ""
        if has_fpn_efficientnet:
            model_type = "FPN_EfficientNet"
        elif has_fcn_resnet:
            model_type = "FCN_ResNet"
        elif has_unet_mobilenet:
            model_type = "UNET_MobileNetV3"
        elif has_unet_resnet:
            model_type = "UNET_ResNet"

        # Initialize dataset with transform matching the model type
        transform = self._get_transform_for_model_type(model_type)
        test_dataset = SegmentationDataset(
            self.config.data_root,
            split="test",
            transform=transform,
            model_type=model_type,
        )

        return DataLoader(
            test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0 if self.device.type == "mps" else self.config.num_workers,
        )

    @staticmethod
    def get_model_type_from_path(model_file: str) -> str:
        """Extract model type from model filename"""
        # Use lowercase comparison to avoid case sensitivity issues
        model_file_lower = model_file.lower()

        if "unet_mobilenet" in model_file_lower:
            return "UNET_MobileNetV3"
        elif "unet_resnet" in model_file_lower:
            return "UNET_ResNet"
        elif "fpn_efficientnet" in model_file_lower:
            return "FPN_EfficientNet"
        elif "fcn_resnet" in model_file_lower:
            return "FCN_ResNet"
        elif "deeplabv3_mobilenet" in model_file_lower:
            return "DeepLabV3_MobileNet"
        elif "deeplabv3_resnet" in model_file_lower:
            return "DeepLabV3_ResNet"
        else:
            # Default to DeepLabV3_ResNet if not recognized
            return "DeepLabV3_ResNet"

    def _initialize_model(self, num_classes: int, model_type: str) -> nn.Module:
        """Initialize model with correct architecture based on model type"""
        if model_type == "DeepLabV3_ResNet":
            model = deeplabv3_resnet50(weights=None)
            model.classifier[-1] = nn.Conv2d(256, num_classes, kernel_size=1)
        elif model_type == "DeepLabV3_MobileNet":
            model = deeplabv3_mobilenet_v3_large(weights=None)
            model.classifier[-1] = nn.Conv2d(256, num_classes, kernel_size=1)
        elif model_type == "FPN_EfficientNet":
            # Initialize FPN with EfficientNet-B0 backbone
            model = smp.FPN(
                "efficientnet-b0",
                encoder_weights=None,  # We're loading weights later
                classes=num_classes,
                activation=None,
            )
            model.segmentation_head = nn.Sequential(
                nn.Conv2d(128, num_classes, kernel_size=1, stride=1),
                nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True),
            )
        elif model_type == "FCN_ResNet":
            # Initialize FCN with ResNet-50 backbone
            model = fcn_resnet50(weights=None)
            model.classifier[-1] = nn.Conv2d(512, num_classes, kernel_size=1, stride=1)
        elif model_type == "UNET_MobileNetV3":
            # Initialize UNet with MobileNetV3-Small backbone
            model = smp.Unet(
                encoder_name="timm-mobilenetv3_small_100",
                encoder_weights=None,  # We're loading weights later
                in_channels=3,
                classes=num_classes,
                activation=None,
            )
        elif model_type == "UNET_ResNet":
            # Initialize UNet with ResNet-34 backbone
            model = smp.Unet(
                encoder_name="resnet34",
                encoder_weights=None,  # We're loading weights later
                in_channels=3,
                classes=num_classes,
                activation=None,
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        return model

    def evaluate_model(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        model_name: str,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Evaluate model performance and generate visualizations"""
        model.eval()
        num_classes = len(CLASS_MAPPING) + 1

        # Get model type for proper output handling
        model_type = self.get_model_type_from_path(model_name)

        # Setup directories and metrics
        model_vis_dir = os.path.join(self.config.save_dir, model_name)
        os.makedirs(model_vis_dir, exist_ok=True)
        confusion_matrix = torch.zeros(num_classes, num_classes, device=self.device)
        vis_count = 0

        # Evaluation loop
        with torch.no_grad():
            with tqdm(total=len(test_loader), desc=f"Evaluating {model_name}") as pbar:
                for images, masks in test_loader:
                    images = images.to(self.device)
                    masks = masks.to(self.device)

                    # Get predictions with model-specific handling
                    outputs = model(images)
                    if model_type in [
                        "DeepLabV3_ResNet",
                        "DeepLabV3_MobileNet",
                        "FCN_ResNet",
                    ]:
                        predictions = torch.argmax(outputs["out"], dim=1)
                    else:  # FPN_EfficientNet, UNET_MobileNetV3, UNET_ResNet
                        predictions = torch.argmax(outputs, dim=1)

                    # Update confusion matrix
                    pred_flat = predictions.view(-1)
                    mask_flat = masks.view(-1)
                    for t, p in [
                        (t, p) for t in range(num_classes) for p in range(num_classes)
                    ]:
                        confusion_matrix[t, p] += torch.sum(
                            (mask_flat == t) & (pred_flat == p)
                        )

                    # Generate visualizations
                    if vis_count < self.config.max_vis_images:
                        for i in range(
                            min(images.size(0), self.config.max_vis_images - vis_count)
                        ):
                            self.visualizer.save_prediction(
                                images[i],
                                masks[i],
                                predictions[i],
                                os.path.join(
                                    model_vis_dir, f"result_img{vis_count}.png"
                                ),
                            )
                            vis_count += 1

                    pbar.update(1)

        return self._calculate_metrics(confusion_matrix, num_classes)

    def _calculate_metrics(
        self, confusion_matrix: torch.Tensor, num_classes: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate IoU and accuracy metrics from confusion matrix"""
        per_class_iou = []
        per_class_acc = []

        for c in range(num_classes):
            intersection = confusion_matrix[c, c]
            gt = confusion_matrix[c, :].sum()
            pred = confusion_matrix[:, c].sum()
            union = gt + pred - intersection

            iou = intersection / (union + 1e-10)
            acc = intersection / (gt + 1e-10)

            per_class_iou.append(iou.item())
            per_class_acc.append(acc.item())

        return np.array(per_class_iou), np.array(per_class_acc)

    def evaluate_all_models(self) -> None:
        """Evaluate all pruned models in the specified directory"""
        test_loader = self._get_test_loader()
        print(f"Loaded {len(cast(Sized, test_loader.dataset))} test images")

        # Get all pruned models and sort them by pruning ratio
        pruned_models = self._get_sorted_pruned_models()

        if not pruned_models:
            print(f"No pruned models found in {self.config.model_dir}")
            return

        print(f"Found {len(pruned_models)} pruned models to evaluate")

        # Evaluate each model
        for model_file, pruning_ratio in pruned_models:
            try:
                # Detect model type from filename
                model_type = self.get_model_type_from_path(model_file)
                print(f"Detected model type: {model_type}")

                # Determine full model path
                model_path = os.path.join(self.config.model_dir, model_file)

                # Check if this is a full model file
                is_full_model = model_file.endswith(".model.pth")

                if is_full_model:
                    print(f"Loading full model from {model_path}")
                    # Load the complete model directly
                    model = torch.load(model_path, map_location=self.device)
                else:
                    # Initialize model with correct architecture
                    num_classes = len(CLASS_MAPPING) + 1
                    model = self._initialize_model(num_classes, model_type)

                    # Load state dict with error handling
                    loaded_state_dict = torch.load(model_path, map_location=self.device)
                    try:
                        tp.load_state_dict(model, state_dict=loaded_state_dict)
                        print("Successfully loaded model using tp.load_state_dict")
                    except Exception as e:
                        print(f"Error using tp.load_state_dict: {e}")
                        # Check if there's an equivalent full model file
                        full_model_path = model_path.replace(".pth", ".model.pth")
                        if os.path.exists(full_model_path):
                            print(
                                f"Found full model at {full_model_path}, loading instead"
                            )
                            model = torch.load(
                                full_model_path, map_location=self.device
                            )
                        elif "full_state_dict" in loaded_state_dict:
                            print("Using 'full_state_dict' to load model")
                            model.load_state_dict(loaded_state_dict["full_state_dict"])
                        else:
                            # Try to clean and convert the state dict for compatibility
                            clean_state_dict = {
                                k: v
                                for k, v in loaded_state_dict.items()
                                if k in model.state_dict()
                                and loaded_state_dict[k].shape
                                == model.state_dict()[k].shape
                            }
                            if not clean_state_dict:
                                print(
                                    "No compatible parameters found in state dict. Model will use random weights."
                                )
                            else:
                                print(
                                    f"Loading {len(clean_state_dict)}/{len(model.state_dict())} compatible parameters"
                                )
                                model.load_state_dict(clean_state_dict, strict=False)

                model.to(self.device)
                print(f"\nEvaluating model: {model_file}")

                # Evaluate model
                mean_iou_per_class, per_class_acc = self.evaluate_model(
                    model,
                    test_loader,
                    model_name=(
                        os.path.splitext(model_file)[0]
                        if not is_full_model
                        else model_file.replace(".model.pth", "")
                    ),
                )

                # Save results with consistent model name (without extension)
                base_model_file = model_file.replace(".model.pth", ".pth")
                self._collect_results(
                    base_model_file, pruning_ratio, mean_iou_per_class, per_class_acc
                )

            except Exception as e:
                print(f"Error evaluating model {model_file}: {str(e)}")
                import traceback

                traceback.print_exc()
                continue

        # Save all results to CSV
        self._save_all_results()
        print("\nEvaluation completed. Run visualization script to generate plots.")

    def _get_sorted_pruned_models(self) -> List[Tuple[str, float]]:
        """Get pruned models sorted by pruning ratio, including both .pth and .model.pth files"""
        pruned_models = []
        model_files = {}  # Dictionary to store file name without extension -> full file name

        # First collect all model files
        for f in os.listdir(self.config.model_dir):
            if f.startswith("pruned_") and (
                f.endswith(".pth") or f.endswith(".model.pth")
            ):
                # Remove extension to get base name
                base_name = f.replace(".model.pth", "").replace(".pth", "")
                # Store or update the file name
                if base_name not in model_files or f.endswith(".model.pth"):
                    # Prefer .model.pth over .pth if both exist
                    model_files[base_name] = f

        # Process model files to extract pruning ratio
        for f in model_files.values():
            if ".model.pth" in f:
                # For .model.pth files, extract ratio from filename
                base_name = f.replace(".model.pth", "")
            else:
                # For standard .pth files
                base_name = f.replace(".pth", "")

            ratio = self._extract_pruning_ratio(base_name)
            if ratio is not None:
                pruned_models.append((f, ratio))

        return sorted(pruned_models, key=lambda x: x[1])

    @staticmethod
    def _extract_pruning_ratio(filename: str) -> Optional[float]:
        """Extract pruning ratio from model filename"""
        match = re.search(r"magnitude_pruner_(0\.\d+)", filename)
        if match:
            return float(match.group(1))
        return None

    def _collect_results(
        self,
        model_file: str,
        pruning_ratio: float,
        mean_iou_per_class: np.ndarray,
        per_class_acc: np.ndarray,
    ) -> None:
        """Collect evaluation results for a single model"""
        class_names = ["background"] + list(CLASS_MAPPING.keys())

        # Create results dictionary
        results = {
            "model_name": model_file,
            "pruning_ratio": pruning_ratio,
            "mean_iou": mean_iou_per_class.mean(),
        }

        # Add per-class metrics
        for i, class_name in enumerate(class_names):
            results[f"iou_{class_name}"] = mean_iou_per_class[i]
            results[f"acc_{class_name}"] = per_class_acc[i]

        self.results_data.append(results)

        # Print current results
        print(f"\nEvaluation results for {model_file}:")
        print(f"Pruning Ratio: {pruning_ratio:.2f}")
        print(f"Mean IoU: {mean_iou_per_class.mean():.4f}")
        for i, class_name in enumerate(class_names):
            print(
                f"{class_name:15s} - IoU: {mean_iou_per_class[i]:.4f}, "
                f"Acc: {per_class_acc[i]:.4f}"
            )
        print("\n--------------------------------\n")

    def _save_all_results(self) -> None:
        """Save all evaluation results to a single CSV file"""
        if not self.results_data:
            return

        df = pd.DataFrame(self.results_data)
        output_file = "evaluation_results_all_models.csv"
        df.to_csv(output_file, index=False)
        print(f"\nAll evaluation results saved to {output_file}")

    def _get_transform_for_model_type(self, model_type: str):
        """Get appropriate transform based on model type"""
        # Choose input size based on model requirements
        if model_type in ["FPN_EfficientNet", "UNET_MobileNetV3", "UNET_ResNet"]:
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


if __name__ == "__main__":
    config = EvaluationConfig()
    evaluator = ModelEvaluator(config)
    evaluator.evaluate_all_models()
