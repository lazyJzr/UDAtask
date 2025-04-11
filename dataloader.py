# -*- coding: utf-8 -*-
import numpy as np
import os
from torch.utils.data import TensorDataset,Dataset, Subset, DataLoader
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms
import torch
from PIL import Image
from utils import parse_args, compute_accuracy

args = parse_args()


augmentations = transforms.Compose([
    transforms.RandomHorizontalFlip(),  
    transforms.RandomVerticalFlip(),    
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2), 
    transforms.ToTensor(),  
])

class AugmentedDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        img = self.data[idx]
        label = self.labels[idx]
        img = Image.fromarray(img.astype(np.uint8).transpose(1, 2, 0))  
            
        if self.transform:
            img = self.transform(img)  

        return img, label
    
def read_directory(directory_name, normal=1):
    file_list = [f for f in os.listdir(directory_name) if not f.startswith('.')]
    file_list.sort(key=lambda x: int(x.split('-')[0]))
    img = []
    label0 = []

    for each_file in file_list:
        img0 = Image.open(directory_name + '/' + each_file)
        gray = img0.resize((args.height, args.width))
        img.append(np.array(gray).astype(float))
        label0.append(float(each_file.split('.')[0].split('-')[1])-1)
    if normal:
        data = np.array(img) / 255.0
    else:
        data = np.array(img)
    data = data.reshape(-1, 3, height, width)
    label = np.array(label0)
    return data, label

def data_loader(path):
    # ../../data/cwt_images/
    data, labels = read_directory(path, normal=1)
    data = torch.tensor(data).type(torch.FloatTensor)
    labels = torch.tensor(labels).type(torch.LongTensor)
    
    dataset = AugmentedDataset(data, labels, transform=augmentations)
    
    train_indices, test_indices = train_test_split(
        range(len(dataset)), test_size=0.2, random_state=42
    )

    train_subset = Subset(dataset, train_indices)
    test_subset = Subset(dataset, test_indices)

    train_loader = torch.utils.data.DataLoader(train_subset, batch_size=24, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_subset, batch_size=24, shuffle=True)

    return train_loader, test_loader

if __name__ == '__main__':
    # data,label = read_directory('../zichai/')
    # print(data.shape)
    # np.set_printoptions(threshold=np.inf)
    # print(label)

    train_loader, test_loader = data_loader('../zichai/')
    train_samples = len(train_loader.dataset)
    test_samples = len(test_loader.dataset)

    print(train_samples, test_samples)





