import json
import os
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch_pruning as tp
import torchvision.transforms as transforms
from PIL import Image
from torchvision.models.segmentation import deeplabv3_resnet50

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
    model_path="../../../models/pruned_models/pruned_Deeplabv3_resnet_epoch_36_magnitude_pruner_0.50.pth",
):
    """Load a pruned DeepLabV3 ResNet model"""
    # Initialize the original model architecture
    model = deeplabv3_resnet50(weights=None)
    model.classifier[-1] = nn.Conv2d(
        256, len(CLASS_MAPPING) + 1, kernel_size=(1, 1), stride=(1, 1)
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
        output_tensor = outputs["out"]

        # Process the output tensor to get predicted classes and probabilities
        probabilities = torch.nn.functional.softmax(output_tensor, dim=1)
        predicted_classes = torch.argmax(probabilities, dim=1)

        # Extract the predicted mask
        predicted_mask = predicted_classes[0].cpu().numpy()

        # Convert mask to RGB for visualization
        color_mask = mask_to_color(predicted_mask)

        # Find the most common class (excluding background class 0)
        classes, counts = np.unique(predicted_mask, return_counts=True)
        non_bg_classes = classes[classes > 0]

        if len(non_bg_classes) > 0:
            non_bg_counts = counts[classes > 0]
            main_class_idx = non_bg_classes[np.argmax(non_bg_counts)]
            main_class_confidence = float(np.max(non_bg_counts)) / float(
                predicted_mask.size
            )

            # Map predicted index to label
            if main_class_idx < len(class_labels):
                predicted_label = class_labels[main_class_idx]
            else:
                predicted_label = "Unknown"

            print(f"Predicted class: {predicted_label}")
            print(f"Confidence: {main_class_confidence:.4f}")

            # Save results if confidence is high enough
            if main_class_confidence > 0.05:  # Lower threshold for pruned model
                # Append predicted label and confidence to a file
                with open("results/predicted_labels_and_confidence.txt", "a") as file:
                    file.write(f"{predicted_label}, {main_class_confidence:.4f}\n")

                # Save the image with its mask
                save_path = "results/detected_images"
                os.makedirs(save_path, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

                # Save original image
                original_image.save(
                    os.path.join(
                        save_path,
                        f"{predicted_label}_original_{main_class_confidence:.4f}_{timestamp}.jpg",
                    )
                )

                # Save mask as image
                mask_image = Image.fromarray(color_mask)
                mask_image.save(
                    os.path.join(
                        save_path,
                        f"{predicted_label}_mask_{main_class_confidence:.4f}_{timestamp}.png",
                    )
                )
            else:
                print("Low confidence. Consider as background.")
        else:
            print("No foreground class detected.")


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
        while True:
            captured_path = capture_image()  # Capture or use existing image
            process_image(captured_path)  # Process the image

            # Wait for user input to continue or exit
            user_input = input("Press Enter to continue or 'q' to quit: ")
            if user_input.lower() == "q":
                break

    except KeyboardInterrupt:
        print("Stopped by User")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    print(f"Running inference with pruned DeepLabV3 ResNet model on {device}")
    print(f"Class labels: {class_labels}")

    # Start the main image processing loop
    main_loop()
