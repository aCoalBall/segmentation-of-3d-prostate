import numpy as np
import torch
import os
from model import UNet3D
import matplotlib.pyplot as plt
import nibabel as nib
import torchio as tio
import torchvision.transforms as transforms

os.chdir(os.path.dirname(__file__))


device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = UNet3D().to(device)
model.load_state_dict(torch.load('net_paras.pth', map_location=device))
model.eval()

image_path = 'v1/semantic_MRs_anon/Case_004_Week0_LFOV.nii.gz'

image = nib.load(image_path)
image = np.asarray(image.dataobj)

totensor = transforms.ToTensor()
image = totensor(image)
print(image.size())

image = image.unsqueeze(0)

#shrink = tio.CropOrPad((16,32,32))
#image = shrink(image)

image = image.unsqueeze(0)
image = image.float().to(device)

pred = model(image)
print(pred.size())
pred = pred.argmax(1)
pred = pred.squeeze(0)
print(pred.size())
pred = torch.permute(pred, (1,2,0))
print(pred.size())

image = image.squeeze()
print(image.size())
image = torch.permute(image, (1,2,0))

image = image.cpu()
pred = pred.cpu()

pred = nib.Nifti1Image(pred.numpy().astype('int16'), None)
img = nib.Nifti1Image(image.numpy().astype('int16'), None)
nib.save(pred, 'sample.nii.gz')
nib.save(img, 'img.nii.gz')



