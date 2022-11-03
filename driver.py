import nibabel as nib
import random
import os
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
import glob
from PIL import Image

import torchio as tio

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import torchvision.transforms as transforms
from model import UNet3D

'''
Class for calculating Dice Similarity Coefficient
'''
class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    '''
    calculate dsc per label
    '''
    def single_loss(self, inputs, targets, smooth=0.1):
        intersection = (inputs * targets).sum()                            
        dice = (2.* intersection + smooth) / (inputs.sum() + targets.sum() + smooth)  
        return dice

    '''
    calculate dsc for each channel, add them up and get the mean
    '''
    def forward(self, inputs, targets, smooth=0.1):    
        
        input0 = (inputs.argmax(1) == 0) # prediction of label 0
        input1 = (inputs.argmax(1) == 1) # prediction of label 1
        input2 = (inputs.argmax(1) == 2) # prediction of label 2
        input3 = (inputs.argmax(1) == 3) # prediction of label 3
        input4 = (inputs.argmax(1) == 4) # prediction of label 4
        input5 = (inputs.argmax(1) == 5) # prediction of label 5

        target0 = (targets == 0) # target of label 0
        target1 = (targets == 1) # target of label 1
        target2 = (targets == 2) # target of label 2
        target3 = (targets == 3) # target of label 3
        target4 = (targets == 4) # target of label 4
        target5 = (targets == 5) # target of label 5
        
        dice0 = self.single_loss(input0, target0)
        dice1 = self.single_loss(input1, target1)
        dice2 = self.single_loss(input2, target2)
        dice3 = self.single_loss(input3, target3)
        dice4 = self.single_loss(input4, target4)
        dice5 = self.single_loss(input5, target5)
        
        dice = (dice0 + dice1 + dice2 + dice3 + dice4 + dice5) / 6.0    
        
        return 1 - dice

'''
Class for loading data from nii.gz files and prepocessing the data
'''
class NiiImageLoader(DataLoader) :
    def __init__(self, image_path, mask_path):
        self.inputs = []
        self.masks = []
        #retrieve path from dataset
        for f in sorted(glob.iglob(image_path)): 
            self.inputs.append(f)

        for f in sorted(glob.iglob(mask_path)):
            self.masks.append(f)

        self.totensor = transforms.ToTensor()

    def __len__(self):
        return len(self.inputs)
    
    #open files
    def __getitem__(self, idx): 
        image_p = self.inputs[idx]
        mask_p = self.masks[idx]

        image = nib.load(image_p)
        image = np.asarray(image.dataobj)

        mask = nib.load(mask_p)
        mask = np.asarray(mask.dataobj)
        
        #Resize the images
        image = self.totensor(image)
        image = image.unsqueeze(0)
        image = image.data

        mask = self.totensor(mask)
        mask = mask.unsqueeze(0)
        mask = mask.data
        
        return image, mask

'''Class for data augmentation'''
class Augment:
    def __init__(self) :
        self.shrink = tio.CropOrPad((16,32,32))
        self.flip0 = tio.transforms.RandomFlip(0, flip_probability = 1) #flip the data randomly
        self.flip1 = tio.transforms.RandomFlip(1, flip_probability = 1)
        self.flip2 = tio.transforms.RandomFlip(2, flip_probability = 1)

        nothing = tio.transforms.RandomFlip(2, flip_probability = 0)
        bias_field = tio.transforms.RandomBiasField()
        blur = tio.transforms.RandomBlur()
        spike = tio.transforms.RandomSpike()
        self.oneof = tio.transforms.OneOf([nothing, bias_field, blur, spike]) #randomly choose one augment method from the three 

    def crop_and_augment(self, image, mask):

        seed = random.randint(0,2)
        if seed == 0:
            image = self.flip0(image)
            mask = self.flip0(mask)
        elif seed == 1:
            image = self.flip1(image)
            mask = self.flip1(mask)
        elif seed == 2:
            image = self.flip2(image)
            mask = self.flip2(mask)

        image = self.oneof(image)
        
        return image, mask


'''
The training function, will train the model save it as 'net_paras.pth'
Loss Function : CrossEntropyLoss
Optimizer : Adam
Epoch : 100
Batch_size : 1
'''

def main() :
    #change path to current directory
    os.chdir(os.path.dirname(__file__))

    #check whether gpu is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device, flush = True)
    torch.cuda.empty_cache()
    #build model and optimizer
    model = UNet3D().to(device)
    loss_fn = nn.CrossEntropyLoss().to(device)
    dice_loss = DiceLoss()
    optimizer = torch.optim.Adam(model.parameters())

    ag = Augment()

    epoch = 30
    loss_list = []
    valid_dsc_list = []
    test_dsc_list = []
    #load the dataset
    dataset = NiiImageLoader("v1/semantic_MRs_anon/*", 
                      "v1/semantic_labels_anon/*")

    #split the dataset
    trainloader, valloader, testloader = torch.utils.data.random_split(dataset, [179, 16, 16])


    #main train loop
    for i in range(epoch) :
        model.train()
        for index, data in enumerate(trainloader, 0) :
            image, mask = data
            image, mask = ag.crop_and_augment(image, mask)
            image = image.unsqueeze(0)
            image = image.float().to(device)
            mask = mask.long().to(device)
            optimizer.zero_grad()
            pred = model(image)
            loss = loss_fn(pred, mask)
            loss.backward()
            optimizer.step()

        
        #run the model on the val set after each train loop
        model.eval()
        num_batches = len(valloader)
        val_loss = 0
        dice_all = 0
        with torch.no_grad():
            for X, y in valloader:
                X = X.unsqueeze(0)
                X = X.float().to(device)
                y = y.long().to(device)
                pred = model(X)
                val_loss += loss_fn(pred, y).item()
                dice_all += (1 - dice_loss(pred, y))
        val_loss /= num_batches
        dice_all /= num_batches
        loss_list.append(val_loss)
        valid_dsc_list.append(dice_all)
        print(f"Avg loss: {val_loss:>8f}", flush = True)
        print(f"DSC: {dice_all:>8f} \n", flush = True)

        print('One Epoch Finished', flush = True)
        torch.save(model.state_dict(), 'net_paras.pth')

        #run on test set after the train is finished
        model.eval()
        num_batches = len(testloader)
        dice_all = 0

        with torch.no_grad():
            for X, y in testloader:
                X = X.unsqueeze(0)
                X = X.float().to(device)
                y = y.long().to(device)
                pred = model(X)
                dice_all += (1 - dice_loss(pred, y))
        dice_all = dice_all / num_batches
        test_dsc_list.append(dice_all)
        print(f"Dice: \n DSC: {dice_all:>8f} \n", flush = True)
    
    np.save('valid_loss.npy', loss_list)
    np.save('valid.npy', valid_dsc_list)
    np.save('test.npy', test_dsc_list)

if __name__ == "__main__":
    main()
