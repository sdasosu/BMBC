import time
import os
import warnings
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from datetime import datetime
import json
import segmentation_models_pytorch as smp

# Suppress TracerWarning from torch.jit if any warnings occur
from torch.jit import TracerWarning

warnings.filterwarnings("ignore", category=TracerWarning)

# ----------------- Setting up the device --------------
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

# ---------------- Read the class labels from a JSON file ---------------
with open("../data/label.json", "r") as f:
    class_labels_dict = json.load(f)
# Sorted keys numerically to ensure the correct order
class_labels = [
    class_labels_dict[key] for key in sorted(class_labels_dict, key=lambda x: int(x))
]
# ------------------------------------------------------------------------

# -------------------- Model ---------------------------
# Build the UNet model using ResNet-34 encoder
model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=3,
    classes=5,  # 4 target classes + background
    activation=None,  # Outputs raw logits
)
model = model.to(device)

# -------------------- Check point --------------------
checkpoint_path = "../models/UNET_resnet_epoch_20.pth"
# Use weights_only=True to limit arbitrary code execution during unpickling if supported
checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
model.load_state_dict(checkpoint, strict=False)
model.eval()  # Set model to evaluation mode
# ----------------------------------------------------------------------------

# For inference acceleration, trace the model with a sample input.
preprocess = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# Use captured.jpg as sample input for tracing instead of a random tensor
try:
    sample_image = Image.open("../data/captured.jpg").convert("RGB")
except Exception as e:
    raise FileNotFoundError("captured.jpg not found for tracing.") from e
tracing_input = preprocess(sample_image).unsqueeze(0).to(device)
net = torch.jit.trace(model, tracing_input, strict=False)
net.eval()


def capture_image(image_path="../data/captured.jpg"):
    """
    Simulate capturing an image.
    In practice, integrate your camera capture command here (e.g., using libcamera-still).
    """
    return image_path


# Decorator to measure inference time (CPU time and wall-clock time)
def measure_inference_time(func):
    def wrapper(*args, **kwargs):
        start_cpu_time = time.process_time()  # Start CPU time measurement
        start_wall_time = time.perf_counter()  # Start wall-clock time measurement

        result = func(*args, **kwargs)

        end_cpu_time = time.process_time()  # End CPU time measurement
        end_wall_time = time.perf_counter()  # End wall-clock time measurement

        cpu_inference_time = end_cpu_time - start_cpu_time
        wall_inference_time = end_wall_time - start_wall_time

        # Get CPU temperature if available
        temp = get_cpu_temperature()

        # Log the inference time and temperature
        log_entry = (
            f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - "
            f"CPU Inference Time: {cpu_inference_time:.6f} sec, "
            f"Wall Inference Time: {wall_inference_time:.6f} sec, "
            f"Temperature: {temp if temp is not None else 'read failed'}"
        )
        with open("results/inference_and_temperature_log.txt", "a") as file:
            file.write(log_entry + "\n")
        print(log_entry)
        return result

    return wrapper


# Process a given image: run inference, log result and save detection if needed
@measure_inference_time
def process_image(image_path):
    # Load and preprocess the image
    image = Image.open(image_path).convert("RGB")
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0).to(
        device
    )  # Create a mini-batch for the model

    with torch.no_grad():
        # Run inference using the traced model
        output_tensor = net(input_batch)
        # Apply softmax to get class probabilities (across channel dimension)
        probabilities = torch.nn.functional.softmax(output_tensor, dim=1)
        # For segmentation, compute the pixel-wise prediction then determine the dominant class
        predicted_classes = torch.argmax(probabilities, dim=1)
        predicted_class = predicted_classes.flatten().mode()[0].item()

        # Map predicted index to label (if within the range of class_labels)
        if predicted_class < len(class_labels):
            predicted_label = class_labels[predicted_class]
        else:
            predicted_label = "Unknown"

        confidence = probabilities.max().item()

        print(f"Predicted class: {predicted_label}")
        print(f"Confidence: {confidence:.4f}")

        if confidence < 0.75:  # Confidence threshold (adjust if needed)
            print("Low confidence. No class detected.")
            return

        # Append predicted label and confidence to a log file
        with open("results/predicted_labels_and_confidence.txt", "a") as file:
            file.write(f"{predicted_label}, {confidence:.4f}\n")

        # Save the image if the predicted label is among specified classes
        if predicted_label in [
            "Others",
            "adult",
            "egg masses",
            "instar nymph (1-3)",
            "instar nymph (4)",
        ]:
            save_path = "results/detected_images"
            os.makedirs(save_path, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            image.save(
                os.path.join(
                    save_path, f"{predicted_label}_{confidence:.4f}_{timestamp}.jpg"
                )
            )


def get_cpu_temperature():
    try:
        temp_str = os.popen("vcgencmd measure_temp").readline()
        temp = float(temp_str.replace("temp=", "").replace("'C\n", ""))
        return temp
    except Exception as e:
        print(f"Could not get temperature: {e}")
        return None


def main_loop():
    try:
        while True:
            capture_image()  # Capture or simulate an image capture
            process_image("../data/captured.jpg")  # Process the captured image
            # Optionally, sleep for a few seconds before the next capture:
            # time.sleep(30)
    except KeyboardInterrupt:
        print("Stopped by User")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    # Start the continuous image capture and inference loop
    main_loop()
