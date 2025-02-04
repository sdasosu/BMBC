import torch
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large
from torchvision.models.segmentation import fcn_resnet50, FCN_ResNet50_Weights
from train import SegmentationDataset, CLASS_MAPPING
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
import matplotlib

matplotlib.use("Agg")  # or "pdf", "svg", etc.
import matplotlib.pyplot as plt
import os
import os
from tqdm import tqdm
import glob
import pandas as pd
import segmentation_models_pytorch as smp

# Define class to color mapping
COLOR_MAP = {
    0: [0, 0, 0],  # Background - black
    1: [255, 0, 0],  # Adult - red
    2: [0, 255, 0],  # egg masses - green
    3: [0, 0, 255],  # instar nymph (1-3) - blue
    4: [255, 255, 0],  # instar nymph (4) - yellow
}


def mask_to_color(mask):
    """Convert class mask to colored image"""
    color_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for class_idx, color in COLOR_MAP.items():
        color_mask[mask == class_idx] = color
    return color_mask


def evaluate_model(model, test_loader, device, save_dir="results", max_vis_images=10):
    model.eval()
    num_classes = (
        len(CLASS_MAPPING) + 1
    )  # Add 1 because there's also a background class
    total_iou = np.zeros(num_classes)
    num_images = 0
    vis_count = 0

    # Initialize confusion matrix
    confusion_matrix = torch.zeros(num_classes, num_classes, device=device)

    # Create save directory
    os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad(), tqdm(total=len(test_loader), desc="Evaluating") as pbar:
        for batch_idx, (images, masks) in enumerate(test_loader):
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            predictions = torch.argmax(outputs, dim=1)

            # Update confusion matrix
            pred_flat = predictions.flatten()
            mask_flat = masks.flatten()
            for t in range(num_classes):
                for p in range(num_classes):
                    confusion_matrix[t, p] += torch.sum(
                        (mask_flat == t) & (pred_flat == p)
                    )

            # Calculate IoU for each class
            for class_idx in range(num_classes):
                class_pred = predictions == class_idx
                class_mask = masks == class_idx
                intersection = torch.logical_and(class_pred, class_mask)
                union = torch.logical_or(class_pred, class_mask)
                iou = torch.sum(intersection) / (torch.sum(union) + 1e-10)
                total_iou[class_idx] += iou.item()

            num_images += images.size(0)

            # Visualize results
            for i in range(min(images.size(0), max_vis_images - vis_count)):
                if vis_count >= max_vis_images:
                    break

                plt.figure(figsize=(15, 5))

                # Original image
                plt.subplot(1, 3, 1)
                img = images[i].cpu().permute(1, 2, 0).numpy()
                img = img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]
                img = np.clip(img * 255, 0, 255).astype(np.uint8)
                plt.imshow(img)
                plt.title("Original Image")
                plt.axis("off")

                # Ground truth
                plt.subplot(1, 3, 2)
                mask_colored = mask_to_color(masks[i].cpu().numpy())
                plt.imshow(mask_colored)
                plt.title("Ground Truth")
                plt.axis("off")

                # Prediction
                plt.subplot(1, 3, 3)
                pred_colored = mask_to_color(predictions[i].cpu().numpy())
                plt.imshow(pred_colored)
                plt.title("Prediction")
                plt.axis("off")

                plt.savefig(
                    os.path.join(save_dir, f"result_img{vis_count}.png"),
                    bbox_inches="tight",
                    pad_inches=0.1,
                )
                plt.close()
                vis_count += 1

            pbar.update(1)

    # -----------------------------------------------------------
    # Compute per-class IoU from the final confusion matrix
    # IoU(c) = conf_mat[c, c] / (sum(conf_mat[c, :]) + sum(conf_mat[:, c]) - conf_mat[c, c])
    # -----------------------------------------------------------
    per_class_iou = []
    for c in range(num_classes):
        intersection = confusion_matrix[c, c]
        gt = confusion_matrix[c, :].sum()
        pred = confusion_matrix[:, c].sum()
        union = gt + pred - intersection
        iou = intersection / (union + 1e-10)
        per_class_iou.append(iou.item())

    # Convert to numpy array for convenience
    mean_iou_per_class = np.array(per_class_iou)

    # Compute per-class accuracy: diag / row_sum
    per_class_acc = confusion_matrix.diag() / (confusion_matrix.sum(dim=1) + 1e-10)
    per_class_acc = per_class_acc.cpu().numpy()

    return mean_iou_per_class, per_class_acc


# -----------------------------------------------------------------
def main():
    # Set device
    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    # Load test dataset
    try:
        test_dataset = SegmentationDataset("../data", split="test")
        test_loader = DataLoader(
            test_dataset,
            batch_size=4,
            shuffle=False,
            num_workers=4 if device.type != "mps" else 0,
        )
        print(f"Loaded {len(test_dataset)} test images")
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        return

    # Initialize model
    try:
        model = smp.FPN(
            "efficientnet-b0", encoder_weights="imagenet", classes=5, activation=None
        )

        model.segmentation_head = nn.Sequential(
            nn.Conv2d(
                128, 5, kernel_size=1, stride=1
            ),  # EfficientNet-B0 outputs 128 feature maps
            nn.Upsample(
                scale_factor=4, mode="bilinear", align_corners=True
            ),  # Adjust resolution
        )
        model.to(device)
        print("Model initialized successfully")
    except Exception as e:
        print(f"Error initializing model: {str(e)}")
        return

    # Find all model checkpoints
    checkpoint_files = sorted(
        glob.glob("models/model_epoch_*.pth"),
        key=lambda x: int(x.split("_")[2].split(".")[0]),
    )

    if not checkpoint_files:
        print("No model checkpoints found!")
        return

    # Evaluate each checkpoint and store results
    results = []
    class_names = ["background"] + list(CLASS_MAPPING.keys())

    for checkpoint_file in checkpoint_files:
        epoch_num = int(checkpoint_file.split("_")[2].split(".")[0])
        print(f"\nEvaluating model from epoch {epoch_num}")

        try:
            # Load model weights
            model.load_state_dict(torch.load(checkpoint_file, map_location=device))

            # Evaluate model
            mean_iou_per_class, per_class_acc = evaluate_model(
                model, test_loader, device, save_dir=f"results/epoch_{epoch_num}"
            )

            result = {"epoch": epoch_num}
            # Add IoU and accuracy for each class
            for i, class_name in enumerate(class_names):
                result[f"iou_{class_name}"] = mean_iou_per_class[i]
                result[f"acc_{class_name}"] = per_class_acc[i]
            result["mean_iou"] = mean_iou_per_class.mean()
            results.append(result)

            print(f"Epoch {epoch_num}:")
            print(f"  Mean IoU: {mean_iou_per_class.mean():.4f}")
            for i, class_name in enumerate(class_names):
                print(
                    f"  {class_name:15s} - IoU: {mean_iou_per_class[i]:.4f}, Acc: {per_class_acc[i]:.4f}"
                )

        except Exception as e:
            print(f"Error evaluating epoch {epoch_num}: {str(e)}")
            continue

    # -----------------------------------------------------------
    # Save results to CSV
    # -----------------------------------------------------------

    df = pd.DataFrame(results)
    df.to_csv("evaluation_results.csv", index=False)
    print("\nResults saved to evaluation_results.csv")

    # Plot curves
    epochs = df["epoch"]

    # Plot mean IoU curve
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, df["mean_iou"], "b-", marker="o")
    plt.title("Mean IoU vs. Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Mean IoU")
    plt.grid(True)
    plt.savefig("miou_curve.png")
    plt.close()

    # Plot IoU curves for each class
    plt.figure(figsize=(12, 6))
    for class_name in class_names:
        plt.plot(epochs, df[f"iou_{class_name}"], marker="o", label=class_name)
    plt.title("Per-Class IoU vs. Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("IoU")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("per_class_iou_curves.png")
    plt.close()

    # Plot accuracy curves for each class
    plt.figure(figsize=(12, 6))
    for class_name in class_names:
        plt.plot(epochs, df[f"acc_{class_name}"], marker="o", label=class_name)
    plt.title("Per-Class Accuracy vs. Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("per_class_accuracy_curves.png")
    plt.close()

    # Find and print best epoch
    best_epoch = epochs[df["mean_iou"].argmax()]
    best_miou = df["mean_iou"].max()
    print(f"\nBest performance: Epoch {best_epoch} with Mean IoU {best_miou:.4f}")
    print("\nPer-class performance at best epoch:")
    best_epoch_data = df[df["epoch"] == best_epoch].iloc[0]
    for class_name in class_names:
        print(
            f'  {class_name:15s} - IoU: {best_epoch_data[f"iou_{class_name}"]:.4f}, '
            f'Acc: {best_epoch_data[f"acc_{class_name}"]:.4f}'
        )


if __name__ == "__main__":
    main()
