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


class SegmentationDataset(Dataset):
    def __init__(self, root_dir, split="train", transform=None):
        self.root_dir = os.path.join(root_dir, split)
        self.transform = (
            transform
            if transform
            else transforms.Compose(
                [
                    transforms.Resize((520, 520)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
        )

        self.images = []
        self.masks = []

        # Get all image files
        for filename in os.listdir(self.root_dir):
            if filename.endswith((".jpg", ".jpeg", ".png")):
                xml_file = os.path.splitext(filename)[0] + ".xml"
                if os.path.exists(os.path.join(self.root_dir, xml_file)):
                    self.images.append(filename)
                    self.masks.append(xml_file)

        logger.info(f"Found {len(self.images)} images in {split} set")

    def __len__(self):
        return len(self.images)

    def parse_xml(self, xml_path):
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()

            # Get image size
            size = root.find("size")
            if size is None:
                logger.error(f"No size information found in {xml_path}")
                raise ValueError("Invalid XML format: missing size information")

            width = int(size.find("width").text)
            height = int(size.find("height").text)

            # Create blank mask
            mask = np.zeros((height, width), dtype=np.uint8)

            # For each object, fill the corresponding area on the mask
            for obj in root.findall("object"):
                # Try to get class name, possibly in name or n tag
                class_name = None
                name_tag = obj.find("name")
                if name_tag is not None:
                    class_name = name_tag.text
                else:
                    n_tag = obj.find("n")
                    if n_tag is not None:
                        class_name = n_tag.text

                if class_name is None:
                    logger.warning(f"Object without class name found in {xml_path}")
                    continue

                if class_name not in CLASS_MAPPING:
                    logger.warning(f"Unknown class {class_name} found in {xml_path}")
                    continue

                bbox = obj.find("bndbox")
                if bbox is None:
                    logger.warning(f"Object without bounding box found in {xml_path}")
                    continue

                try:
                    xmin = max(0, int(bbox.find("xmin").text))
                    ymin = max(0, int(bbox.find("ymin").text))
                    xmax = min(width, int(bbox.find("xmax").text))
                    ymax = min(height, int(bbox.find("ymax").text))
                except (ValueError, AttributeError) as e:
                    logger.warning(f"Invalid bounding box values in {xml_path}: {e}")
                    continue

                # Ensure bounding box coordinates are valid
                if xmin >= xmax or ymin >= ymax:
                    logger.warning(f"Invalid bounding box coordinates in {xml_path}")
                    continue

                # Mark the bounding box area with the corresponding class index
                mask[ymin:ymax, xmin:xmax] = CLASS_MAPPING[class_name]

            return Image.fromarray(mask)

        except Exception as e:
            logger.error(f"Error parsing XML file {xml_path}: {e}")
            # Return blank mask instead of throwing an exception
            return Image.fromarray(np.zeros((height, width), dtype=np.uint8))

    def __getitem__(self, idx):
        # Load image
        img_path = os.path.join(self.root_dir, self.images[idx])
        image = Image.open(img_path).convert("RGB")

        # Load and parse XML
        xml_path = os.path.join(self.root_dir, self.masks[idx])
        mask = self.parse_xml(xml_path)

        # Apply transformations
        if self.transform:
            image = self.transform(image)
            mask = transforms.Resize((520, 520))(mask)
            mask = torch.from_numpy(np.array(mask)).long()

        return image, mask
