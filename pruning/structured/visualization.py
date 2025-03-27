import argparse
import os

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Use non-interactive backend
matplotlib.use("Agg")

# Set ACM style fonts and sizes
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "font.size": 12,
        "axes.labelsize": 14,
        "axes.titlesize": 16,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "figure.titlesize": 18,
    }
)

# Set seaborn style for better academic visualizations
sns.set_style("whitegrid")
sns.set_context("paper")

# Color mapping for visualization - copied from inference.py for consistency
COLOR_MAP = {
    0: [0, 0, 0],  # Background - Black
    1: [255, 0, 0],  # Adult - Red
    2: [0, 255, 0],  # Egg masses - Green
    3: [0, 0, 255],  # Instar nymph (1-3) - Blue
    4: [255, 255, 0],  # Instar nymph (4) - Yellow
}


class VisualizationGenerator:
    """Handles generation of visualization from evaluation results"""

    def __init__(self, csv_file: str, save_dir: str = "visualization_results"):
        """
        Initialize visualization generator

        Args:
            csv_file: Path to CSV file with evaluation results
            save_dir: Directory to save visualization results
        """
        self.csv_file = csv_file
        self.save_dir = save_dir
        self.df = None
        self.class_names = []

    def _load_data(self) -> bool:
        """Load data from CSV file"""
        if not os.path.exists(self.csv_file):
            print(f"Error: CSV file not found: {self.csv_file}")
            return False

        try:
            self.df = pd.read_csv(self.csv_file)
            print(f"Loaded data from {self.csv_file}")

            # Extract class names from column names
            self.class_names = [
                col.replace("iou_", "")
                for col in self.df.columns
                if col.startswith("iou_")
            ]
            print(f"Found class names: {self.class_names}")

            # Add model_type column if not present
            if "model_type" not in self.df.columns:
                self.df["model_type"] = self.df["model_name"].apply(
                    self._extract_model_type
                )

            return True
        except Exception as e:
            print(f"Error loading CSV file: {str(e)}")
            return False

    @staticmethod
    def _extract_model_type(model_file: str) -> str:
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

    def generate_visualizations(self) -> bool:
        """Generate all visualizations from loaded data"""
        if not self._load_data() or self.df is None or self.df.empty:
            return False

        # Create visualization directory
        os.makedirs(self.save_dir, exist_ok=True)

        # Generate different visualizations
        self._generate_mean_iou_curves()
        self._generate_per_class_iou_curves()
        self._generate_model_comparison_curves()
        self._generate_per_model_curves()

        return True

    def _generate_mean_iou_curves(self) -> None:
        """Generate mean IoU curves"""
        # Create figure with ACM compatible size - narrower for side-by-side display
        plt.figure(figsize=(5, 4.5))

        # Plot mean IoU vs pruning ratio grouped by model type
        sns.lineplot(
            data=self.df, x="pruning_ratio", y="mean_iou", hue="model_type", marker="o"
        )
        plt.title("Mean IoU vs Pruning Ratio")
        plt.xlabel("Pruning Ratio")
        plt.ylabel("Mean IoU")
        plt.grid(True, alpha=0.3)

        # Improve legend readability
        plt.legend(title="Model Type", frameon=True, loc="best")

        # Ensure tick labels are readable
        plt.tight_layout(pad=1.2)

        # Save as both PNG and PDF
        base_path = os.path.join(self.save_dir, "mean_iou_curves")
        plt.savefig(
            f"{base_path}.png",
            bbox_inches="tight",
            dpi=600,
        )
        plt.savefig(f"{base_path}.pdf", bbox_inches="tight", format="pdf")
        plt.close()

    def _generate_per_class_iou_curves(self) -> None:
        """Generate per-class IoU curves"""
        # Create figure with ACM compatible size - narrower for side-by-side display
        plt.figure(figsize=(5, 4.5))

        # Plot per-class IoU vs pruning ratio
        for class_name in self.class_names:
            sns.lineplot(
                data=self.df,
                x="pruning_ratio",
                y=f"iou_{class_name}",
                label=class_name,
                marker="o",
            )
        plt.title("Per-class IoU vs Pruning Ratio")
        plt.xlabel("Pruning Ratio")
        plt.ylabel("IoU")

        # Improve legend readability - may need smaller fontsize if many classes
        plt.legend(
            loc="upper right",
            frameon=True,
            fontsize=10 if len(self.class_names) > 4 else 12,
        )
        plt.grid(True, alpha=0.3)

        # Ensure tick labels are readable
        plt.tight_layout(pad=1.2)

        # Save as both PNG and PDF
        base_path = os.path.join(self.save_dir, "per_class_iou_curves")
        plt.savefig(
            f"{base_path}.png",
            bbox_inches="tight",
            dpi=600,
        )
        plt.savefig(f"{base_path}.pdf", bbox_inches="tight", format="pdf")
        plt.close()

    def _generate_model_comparison_curves(self) -> None:
        """Generate model comparison curves for IoU and accuracy"""
        # Create a new figure for model comparison
        plt.figure(figsize=(10, 5))

        # Create 2x1 subplots for IoU and Accuracy
        plt.subplot(1, 2, 1)
        # Group by model type and pruning ratio to calculate average performance
        model_comparison = (
            self.df.groupby(["model_type", "pruning_ratio"])["mean_iou"]
            .mean()
            .reset_index()
        )

        # Plot model comparison - IoU
        sns.lineplot(
            data=model_comparison,
            x="pruning_ratio",
            y="mean_iou",
            hue="model_type",
            marker="o",
        )
        plt.title("Model Comparison: Mean IoU", fontsize=14)
        plt.xlabel("Pruning Ratio", fontsize=12)
        plt.ylabel("Mean IoU", fontsize=12)
        plt.grid(True, alpha=0.3)

        # Improve legend readability
        plt.legend(title="Model Type", fontsize=10, loc="best")

        # Calculate mean accuracy for each model type and pruning ratio
        plt.subplot(1, 2, 2)

        # Calculate mean accuracy for each model if not already present
        if "mean_acc" not in self.df.columns:
            self.df["mean_acc"] = self.df[[f"acc_{c}" for c in self.class_names]].mean(
                axis=1
            )

        model_acc_comparison = (
            self.df.groupby(["model_type", "pruning_ratio"])["mean_acc"]
            .mean()
            .reset_index()
        )

        # Plot model comparison - Accuracy
        sns.lineplot(
            data=model_acc_comparison,
            x="pruning_ratio",
            y="mean_acc",
            hue="model_type",
            marker="o",
        )
        plt.title("Model Comparison: Mean Accuracy", fontsize=14)
        plt.xlabel("Pruning Ratio", fontsize=12)
        plt.ylabel("Mean Accuracy", fontsize=12)
        plt.grid(True, alpha=0.3)

        # Improve legend readability
        plt.legend(title="Model Type", fontsize=10, loc="best")

        # Save model comparison plot in both formats
        plt.tight_layout(pad=1.2)
        base_path = os.path.join(self.save_dir, "model_comparison_curves")
        plt.savefig(
            f"{base_path}.png",
            bbox_inches="tight",
            dpi=600,
        )
        plt.savefig(f"{base_path}.pdf", bbox_inches="tight", format="pdf")
        plt.close()

    def _generate_per_model_curves(self) -> None:
        """Generate per-model IoU curves for each class"""
        # Create separate plots for each model type showing per-class IoU
        unique_model_types = self.df["model_type"].unique()
        num_model_types = len(unique_model_types)

        if num_model_types > 0:
            # Determine grid layout
            if num_model_types <= 2:
                nrows, ncols = 1, num_model_types
            else:
                nrows = (num_model_types + 1) // 2
                ncols = 2

            # Handle special case: when there's only one model type
            if num_model_types == 1:
                _, ax = plt.subplots(figsize=(8, 6))
                axes = [ax]  # Create a list with the single axis
            else:
                _, axes_array = plt.subplots(
                    nrows=nrows, ncols=ncols, figsize=(10, 6 * nrows)
                )
                # Create a list from the axes array
                if nrows == 1 and ncols > 1:
                    axes = list(axes_array)
                elif nrows > 1 and ncols > 1:
                    axes = [ax for row in axes_array for ax in row]
                else:
                    axes = [axes_array]

            # Plot per-class IoU for each model type
            for i, model_type in enumerate(unique_model_types):
                model_df = self.df[self.df["model_type"] == model_type]

                for class_name in self.class_names:
                    sns.lineplot(
                        data=model_df,
                        x="pruning_ratio",
                        y=f"iou_{class_name}",
                        label=class_name,
                        marker="o",
                        ax=axes[i],
                    )

                axes[i].set_title(f"{model_type}: Per-class IoU vs Pruning Ratio")
                axes[i].set_xlabel("Pruning Ratio")
                axes[i].set_ylabel("IoU")
                axes[i].grid(True, alpha=0.3)

                # Adjust legend based on number of classes
                axes[i].legend(
                    loc="best",
                    frameon=True,
                    fontsize=10 if len(self.class_names) > 4 else 12,
                )

            # Hide any unused subplots (only needed when num_model_types > 1)
            if num_model_types > 1:
                for j in range(i + 1, len(axes)):
                    axes[j].set_visible(False)

            # Save per-model plots in both formats
            plt.tight_layout(pad=1.2)
            base_path = os.path.join(self.save_dir, "per_model_class_iou")
            plt.savefig(
                f"{base_path}.png",
                bbox_inches="tight",
                dpi=600,
            )
            plt.savefig(f"{base_path}.pdf", bbox_inches="tight", format="pdf")
            plt.close()

        # Print summary of generated visualizations
        print(f"\nVisualization results saved to {self.save_dir}/ directory:")
        print(
            f"- Mean IoU curves: PNG: {self.save_dir}/mean_iou_curves.png | PDF: {self.save_dir}/mean_iou_curves.pdf"
        )
        print(
            f"- Per-class IoU curves: PNG: {self.save_dir}/per_class_iou_curves.png | PDF: {self.save_dir}/per_class_iou_curves.pdf"
        )
        print(
            f"- Model comparison curves: PNG: {self.save_dir}/model_comparison_curves.png | PDF: {self.save_dir}/model_comparison_curves.pdf"
        )
        if num_model_types > 0:
            print(
                f"- Per-model class IoU curves: PNG: {self.save_dir}/per_model_class_iou.png | PDF: {self.save_dir}/per_model_class_iou.pdf"
            )


def main():
    """Main function to generate visualizations from CSV file"""
    parser = argparse.ArgumentParser(
        description="Generate visualizations from model evaluation results"
    )
    parser.add_argument(
        "--csv-file",
        type=str,
        default="evaluation_results_all_models.csv",
        help="Path to CSV file with evaluation results",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="visualization_results",
        help="Directory to save visualization results",
    )
    args = parser.parse_args()

    visualizer = VisualizationGenerator(args.csv_file, args.save_dir)
    success = visualizer.generate_visualizations()

    if success:
        print("Visualization generation completed successfully!")
    else:
        print("Failed to generate visualizations.")
        return 1

    return 0


if __name__ == "__main__":
    main()
