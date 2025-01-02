import os
import torch
import xml.etree.ElementTree as ET
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np

#----------------------------------------------------------
def find_classes(train_dir):
    classes_set=set()

    for file in os.listdir(train_dir):
        if file.lower().endswith('xml'):
            xml_path=os.path.join(train_dir,file)
            tree=ET.parse(xml_path)
            root=tree.getroot()

            for obj in root.findall('object'):
                class_name=obj.find('name').text.strip()
                classes_set.add(class_name)
                

