import io
import json
import os
import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch_pruning as tp
import torchvision.transforms as transforms
from PIL import Image
from torchvision.models.segmentation import fcn_resnet50

# ----------------- Setting up the device--------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------- Create necessary directories if they don't exist --------------
os.makedirs("results", exist_ok=True)
os.makedirs(os.path.join("results", "detected_images"), exist_ok=True)
# -------------------------------------------------------------------------------

# ----------------- Class Mapping ---------------------
CLASS_MAPPING = {
    "adult": 1,
    "egg masses": 2,
    "instar nymph (1-3)": 3,
    "instar nymph (4)": 4,
}
# ---------------------------------------------------

# ----------------- Color Mapping for Visualization ---------------------
COLOR_MAP = {
    0: [0, 0, 0],  # Background - Black
    1: [255, 0, 0],  # Adult - Red
    2: [0, 255, 0],  # Egg masses - Green
    3: [0, 0, 255],  # Instar nymph (1-3) - Blue
    4: [255, 255, 0],  # Instar nymph (4) - Yellow
}
# ---------------------------------------------------


# ---------------- Read the class labels from a JSON file ---------------
def load_class_labels(label_path="../../../data/label.json"):
    try:
        with open(label_path, "r") as f:
            class_labels_dict = json.load(f)

        # Sorted keys numerically to ensure the correct order
        class_labels = [
            class_labels_dict[key]
            for key in sorted(class_labels_dict, key=lambda x: int(x))
        ]
        return class_labels
    except Exception as e:
        print(f"Error loading class labels: {e}")
        # Fallback class labels if file not found
        return [
            "Background",
            "adult",
            "egg masses",
            "instar nymph (1-3)",
            "instar nymph (4)",
        ]


# ------------------------------------------------------------------------

class_labels = load_class_labels()


# -------------------- Load Pruned Model ---------------------------
def load_pruned_model(
    model_path="../../../models/pruned_models/pruned_FCN_resnet_epoch_19_magnitude_pruner_0.50.pth",
):
    """Load a pruned FCN ResNet model"""
    # Initialize the original model architecture
    model = fcn_resnet50(weights=None)
    model.classifier[-1] = nn.Conv2d(
        512, len(CLASS_MAPPING) + 1, kernel_size=(1, 1), stride=(1, 1)
    )

    try:
        # Load the state dict
        loaded_state_dict = torch.load(model_path, map_location=device)

        # Try using torch_pruning to load the state dict (for structured pruning)
        try:
            tp.load_state_dict(model, state_dict=loaded_state_dict)
            print(
                f"Successfully loaded pruned model using tp.load_state_dict from {model_path}"
            )
        except Exception as e:
            print(f"Error using tp.load_state_dict: {e}")
            print("Falling back to regular state_dict loading")

            # Try regular loading as fallback
            model.load_state_dict(loaded_state_dict, strict=False)
            print(
                f"Successfully loaded pruned model with regular loading from {model_path}"
            )
    except Exception as e:
        print(f"Error loading pruned model: {e}")
        print("Using randomly initialized model instead")

    model = model.to(device)
    model.eval()  # Set model to evaluation mode

    return model


# ----------------------------------------------------------------

# Load pruned model
model = load_pruned_model()

# Define preprocessing transformations for inference
preprocess = transforms.Compose(
    [
        transforms.Resize((520, 520)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def capture_image(image_path="../../../data/captured.jpg"):
    """
    Captures an image using libcamera-still (on Raspberry Pi)
    or simulates capturing by using an existing image
    """
    # Uncomment the following line on Raspberry Pi to capture an image
    # os.system(f'libcamera-still -o {image_path}')

    # For testing/development, verify the file exists
    if not os.path.exists(image_path):
        print(f"Warning: {image_path} does not exist. Using a test image if available.")
        test_images = [
            f
            for f in os.listdir("../../../data")
            if f.endswith((".jpg", ".png", ".jpeg"))
        ]
        if test_images:
            image_path = os.path.join("../../../data", test_images[0])
            print(f"Using test image: {image_path}")

    return image_path


def mask_to_color(mask):
    """Convert a class mask to a color image for visualization"""
    color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)

    for class_idx, color in COLOR_MAP.items():
        color_mask[mask == class_idx] = color

    return color_mask


# Decorator to measure inference time
def measure_inference_time(func):
    def wrapper(*args, **kwargs):
        start_cpu_time = time.process_time()  # Start CPU time measurement
        start_wall_time = time.perf_counter()  # Start wall-clock time measurement

        result = func(*args, **kwargs)

        end_cpu_time = time.process_time()  # End CPU time measurement
        end_wall_time = time.perf_counter()  # End wall-clock time measurement

        cpu_inference_time = end_cpu_time - start_cpu_time
        wall_inference_time = end_wall_time - start_wall_time

        # Get CPU temperature
        temp = get_cpu_temperature()

        # Create a log entry with timestamp, CPU time, wall time, and temperature information
        log_entry = (
            f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - "
            f"CPU Inference Time: {cpu_inference_time:.6f} sec, "
            f"Wall Inference Time: {wall_inference_time:.6f} sec, "
            f"Temperature: {temp if temp is not None else 'read failed'}"
        )

        # Write the log entry to a file and print it
        with open("results/inference_and_temperature_log.txt", "a") as file:
            file.write(log_entry + "\n")
        print(log_entry)

        return result

    return wrapper


@measure_inference_time
def process_image(image_path):
    """Process an image and run inference with the pruned model"""
    # Load and preprocess the image
    image = Image.open(image_path).convert("RGB")
    original_image = image.copy()  # Keep a copy of the original image

    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0).to(device)  # Create a mini-batch

    # Perform inference
    with torch.no_grad():
        outputs = model(input_batch)
        # Check the output type
        if isinstance(outputs, dict):
            output_tensor = outputs["out"]
        else:
            output_tensor = outputs

        # Get softmax probabilities
        probabilities = torch.nn.functional.softmax(output_tensor, dim=1)

        # Sum the probabilities for each class
        class_probabilities_sum = probabilities.sum(dim=[2, 3])[0]

        # Ignore background class (channel 0) and focus on foreground classes
        foreground_probs = class_probabilities_sum[1:]

        # Check if there are any foreground classes with non-zero probability
        if foreground_probs.sum().item() > 0:
            # Normalize the foreground probabilities
            foreground_probs = foreground_probs / foreground_probs.sum()

            # Get the class with highest probability among foreground classes
            # Add 1 to restore original class index (since we ignored background)
            fg_class_idx = foreground_probs.argmax().item()
            main_class_idx = fg_class_idx + 1
            main_class_confidence = foreground_probs[fg_class_idx].item()

            # Set the confidence threshold
            confidence_threshold = 0.05  # Adjust as needed

            if main_class_confidence < confidence_threshold:
                predicted_label = "unknown"
                print(
                    f"Low confidence detection ({main_class_confidence:.4f}). Marking as unknown."
                )
            else:
                if main_class_idx < len(class_labels):
                    predicted_label = class_labels[main_class_idx]
                else:
                    predicted_label = "unknown"

                print(f"Predicted class: {predicted_label}")
                print(f"Confidence: {main_class_confidence:.4f}")
        else:
            # If no foreground class detected, mark as unknown
            predicted_label = "unknown"
            print("No foreground class detected. Marking as unknown.")

        # Generate a predicted mask for visualization
        predicted_classes = torch.argmax(probabilities, dim=1)
        predicted_mask = predicted_classes[0].cpu().numpy()
        color_mask = mask_to_color(predicted_mask)

        # Save the image and mask
        save_path = "results/detected_images"
        os.makedirs(save_path, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save the original image
        original_image.save(
            os.path.join(
                save_path,
                f"{predicted_label}_original_{0 if predicted_label == 'unknown' else main_class_confidence:.4f}_{timestamp}.png",
            )
        )

        # Save the mask
        mask_image = Image.fromarray(color_mask)
        mask_image.save(
            os.path.join(
                save_path,
                f"{predicted_label}_mask_{0 if predicted_label == 'unknown' else main_class_confidence:.4f}_{timestamp}.png",
            )
        )

        # Only record when the prediction is not background or unknown
        if predicted_label not in ["background", "unknown"]:
            with open("results/predicted_labels_and_confidence.txt", "a") as file:
                file.write(f"{predicted_label}, {main_class_confidence:.4f}\n")

        # Create visualization
        comparison_image = create_visualization(
            input_tensor.cpu(),
            mask_image,
            predicted_label,
            main_class_confidence,
        )

        # Save the comparison image
        comparison_image.save(
            os.path.join(
                save_path,
                f"{predicted_label}_comparison_{0 if predicted_label == 'unknown' else main_class_confidence:.4f}_{timestamp}.png",
            )
        )


def create_visualization(preprocessed_tensor, mask_image, predicted_label, confidence):
    """Create a visualization of the preprocessed image and mask"""
    # Inverse normalization - Convert pixel values back to [0,1] range
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])

    # Clone the tensor to avoid modifying the original data
    display_tensor = preprocessed_tensor.clone()

    # Inverse normalization - Convert pixel values back to [0,1] range
    for t, m, s in zip(display_tensor, mean, std):
        t.mul_(s).add_(m)

    # Convert the tensor to a numpy array for display
    display_image = display_tensor.permute(1, 2, 0).numpy()
    display_image = np.clip(display_image, 0, 1)

    # Convert the mask to a numpy array
    mask_image_np = np.array(mask_image)

    # Create an image canvas
    plt.figure(figsize=(12, 6))

    # Left: Processed image
    plt.subplot(1, 2, 1)
    plt.imshow(display_image)
    plt.title("Processed Image")
    plt.axis("off")

    # Right: Prediction mask
    plt.subplot(1, 2, 2)
    plt.imshow(mask_image_np)
    plt.title(f"Prediction: {predicted_label}\nConfidence: {confidence:.4f}")
    plt.axis("off")

    # Save as an image object in memory
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.1)
    plt.close()
    buf.seek(0)

    # Convert to a PIL image
    comparison_img = Image.open(buf)
    comparison_img = comparison_img.convert("RGB")
    return comparison_img


def get_cpu_temperature():
    """Get CPU temperature (works on Raspberry Pi)"""
    try:
        temp_str = os.popen("vcgencmd measure_temp").readline()
        temp = float(temp_str.replace("temp=", "").replace("'C\n", ""))
        return temp
    except Exception as e:
        # This might fail on non-Raspberry Pi systems
        try:
            # Try an alternative method for Linux systems
            if os.path.exists("/sys/class/thermal/thermal_zone0/temp"):
                with open("/sys/class/thermal/thermal_zone0/temp") as f:
                    temp = float(f.read()) / 1000.0
                return temp
        except Exception:
            pass

        print(f"Could not get temperature: {e}")
        return None


def main_loop():
    """Main inference loop"""
    try:
        last_modified_time = 0
        image_path = "../../../data/captured.jpg"

        while True:
            captured_path = capture_image()  # Capture or use existing image

            # Check if the image has been modified
            current_modified_time = os.path.getmtime(image_path)

            if current_modified_time > last_modified_time:
                print(
                    f"New image detected at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                )
                process_image(captured_path)  # Process the captured image
                last_modified_time = current_modified_time

            time.sleep(0.2)  # Check for new image every 200ms

    except KeyboardInterrupt:
        print("\nStopped by User")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    # Start the main image processing loop
    main_loop()
