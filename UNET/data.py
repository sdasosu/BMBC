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
    classes_list = sorted(list(classes_set))
    return classes_list
#----------------------------------------------------------
class SegmentationDataSet(Dataset):
    #--------------------- init ---------------------------
    def __init__(self, data_dir, class_to_idx, transform=None):
        super().__init__()
        self.data_dir=data_dir
        self.class_toidx = class_to_idx
        self.transform = transform

        self.image_files=[]
        for file in os.listdir(data_dir):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                xml_name = os.path.splitext(file)[0] + '.xml'
                xml_path = os.path.join(data_dir, xml_name)
                if os.path.exists(xml_path):
                    self.image_files.append(file)
        self.image_files = sorted(self.image_files)
    #-------------------- len ----------------------------
    def __len__(self):
        return len(self.image_files)

    #-------------------- get item -----------------------
    def __getitem__(self, idx):
        #--------- Load image --------
        image_name=self.image_files[idx]
        image_path=os.path.join(self.data_dir, image_name)
        image=Image.open(image_path).convert('RGB')

        #-------- Perse XML annotation ------------------
        xml_name = os.path.splitext(image_name)[0] + '.xml'
        xml_path = os.path.join(self.data_dir, xml_name)
        width, height = image.size
        mask=np.zeros((height,width), dtype=np.uint8)

        if os.path.exists(xml_path):
            tree=ET.parse(xml_path)
            root=tree.getroot()
            
            for obj in root.findall('object'):
                class_name = obj.find('name').text.strip()
                if class_name not in self.class_to_idx:
                    # If it's not in our known classes, skip
                    continue
                class_idx = self.class_to_idx[class_name]
                bndbox = obj.find('bndbox')
                xmin = int(bndbox.find('xmin').text)
                ymin = int(bndbox.find('ymin').text)
                xmax = int(bndbox.find('xmax').text)
                ymax = int(bndbox.find('ymax').text)
                mask[ymin:ymax, xmin:xmax] = class_idx
            
            
            if self.transform is not None:
                image = self.transform(image)
            else:
                image = transforms.ToTensor()(image)

            mask = torch.from_numpy(mask).long()  # shape: (H, W)
        
        return image, mask


    def get_dataloader(data_dir, class_to_idx, batch_size=4, shuffle=True):
        """
        Utility function to return a DataLoader for the given data_dir.
        """
        dataset = SegmentationDataSet(
            data_dir=data_dir,
            class_to_idx=class_to_idx,
            transform=transforms.ToTensor()  # or more advanced transforms
        )
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        return dataloader       



