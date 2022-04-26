# Imports
import matplotlib.pyplot as plt
import numpy as np
import random
from PIL import Image
import PIL.ImageOps   
import os 

from tqdm import tqdm

import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torchvision.utils
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from pathlib import Path


### VISUALIZATION ###
def imshow(img, text=None):
    """"
        Visualization
    """
    npimg = img.numpy()
    plt.axis("off")
    if text:
        plt.text(75, 8, text, style='italic',fontweight='bold',
            bbox={'facecolor':'white', 'alpha':0.8, 'pad':10})
        
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()    

# Plotting data
def show_plot(iteration,loss):
    """
        Plot the loss by iteration
    """
    plt.plot(iteration,loss)
    plt.show()


### DATASET ###

class SiameseNetworkDataset(Dataset):
    """"
        Dataset that returns pairs of images along with a label stating whether or not they are the same
    """
    def __init__(self,imageFolderDataset,transform=None):
        self.imageFolderDataset = imageFolderDataset    
        self.transform = transform
        
    def __getitem__(self,index):
        img0_tuple = random.choice(self.imageFolderDataset.imgs)

        #We need to approximately 50% of images to be in the same class
        should_get_same_class = random.randint(0,1) 
        if should_get_same_class:
            while True:
                #Look untill the same class image is found
                img1_tuple = random.choice(self.imageFolderDataset.imgs) 
                if img0_tuple[1] == img1_tuple[1]:
                    break
        else:

            while True:
                #Look untill a different class image is found
                img1_tuple = random.choice(self.imageFolderDataset.imgs) 
                if img0_tuple[1] != img1_tuple[1]:
                    break

        img0 = Image.open(img0_tuple[0])
        img1 = Image.open(img1_tuple[0])

        img0 = img0.convert("L")
        img1 = img1.convert("L")

        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)
        
        return img0, img1, torch.from_numpy(np.array([int(img1_tuple[1] != img0_tuple[1])], dtype=np.float32))
    
    def __len__(self):
        return len(self.imageFolderDataset.imgs)

### NETWORK ###

class SiameseNetwork(nn.Module):
    """
        Implementation of the network
    """

    def __init__(self):
        super(SiameseNetwork, self).__init__()

        # Setting up the Sequential of CNN Layers
        self.cnn1 = nn.Sequential(
            nn.Conv2d(1, 96, kernel_size=11,stride=4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),
            
            nn.Conv2d(96, 256, kernel_size=5, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(256, 384, kernel_size=3,stride=1),
            nn.ReLU(inplace=True)
        )

        # Setting up the Fully Connected Layers
        self.fc1 = nn.Sequential(
            nn.Linear(384, 1024),
            nn.ReLU(inplace=True),
            
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            
            nn.Linear(256,2)
        )
        
    def forward_once(self, x):
        # This function will be called for both images
        # Its output is used to determine the similiarity
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output

    def forward(self, input1, input2):
        # In this function we pass in both images and obtain both vectors
        # which are returned
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)

        return output1, output2

### LOSS FUNCTION ###

class ContrastiveLoss(torch.nn.Module):
    """
        Loss is euclidean distance if same class, else margin - euclidean distance
    """
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        # Calculate the euclidean distance and calculate the contrastive loss
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim = True)

        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                    (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))


        return loss_contrastive

### INFERENCE DATASET ###
from PIL import Image
import numpy as np

class SiameseNetworkDatasetInference(Dataset):
    """
        Only returns one image and its class (the name of the person)
    """
    def __init__(self, images, transform=None):
        self.images = images # list of paths
        self.transform = transform
        
    def __getitem__(self,index):
        # get an image path
        input_ID = self.images[index]

        # Load input
        x = Image.open(input_ID)
        if self.transform is not None:
            x = self.transform(x)
            
        if x.shape[0] == 3 :
            x = x[0,:,:]
            x = np.expand_dims(x, axis=0)
            
        cl = Path(input_ID).parts
        
        return x, cl # return the image and the class as str
    
    def __len__(self):
        return len(self.images)
    
def create_dataset_testing(direc : str, transform = None, batch_size=5) :
    """
        Directory contains all the folders (with the name of the person) that have images
    """
    inputs = []
    for directory in os.listdir(direc) :
        for file in os.listdir(os.path.join(direc,directory)) :
            inputs.append(os.path.join(os.path.join(direc,directory),file))
            
    test = SiameseNetworkDatasetInference(inputs, transform=transform)
    test_dataloader = DataLoader(test, num_workers=0, batch_size=batch_size, shuffle=True)
    
    return test_dataloader

def create_dataset_topredict(image : str, transform = None) :
    """
        Create a similar dataset but containing only the image to classify
    """
    image_prediction = SiameseNetworkDatasetInference([image], transform=transform)
    prediction_dataloader = DataLoader(image_prediction, num_workers=0, batch_size=1, shuffle=True)
    
    return prediction_dataloader

def find_best_match(prediction_dataloader, test_dataloader, device, net) :
    """
        Return the value of the score along side with a path to the image
    """
    x0, _ = next(iter(prediction_dataloader))
    x0 = x0.to(device)

    best_score = 100
    best_class = ""

    for x1, cl in tqdm(test_dataloader) :
        # Concatenate the two images together
        output1, output2 = net(x0, x1.to(device))
        euclidean_distance = F.pairwise_distance(output1, output2)
        if torch.min(euclidean_distance) < best_score :
            best_score = torch.min(euclidean_distance)
            best_class = [x[torch.argmin(euclidean_distance)] for x in cl]

    return best_score.item() , best_class