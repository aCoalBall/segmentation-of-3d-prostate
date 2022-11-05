Image Segmentation for prostate dataset using UNet-3D
======
 
 ## UNet-3D
 
 Volumetric data is abundant in biomedical data analysis. UNet-3D is a variant of UNet for volumetric segmentaton that learns from sparsely annotated volumetric images. Instead of using 2D images, each sample in a 3d unet is a volumetric image.
 
 
 
 ## Algorithnm in this folder
 
 This folder contains a 3d unet model which is designed for image segmentation of [Prostate 3D data set](). The model recieves input as size (B x C x H x W x D), where B represents batch size, C represents the number of channels, H, W, D represents height, width and depth respectively. And it outputs a tensor in size (B x 6 x H x W x D), where 6 represents the probability of each label in the given voxel. The detailed architechture of each layer is illustrated in the documentation of codes (model.py).
 
 model.py is the 3d unet model class and driver.py is a script for training a model on the dataset.
 
 
 ## Dependencies
 
 The program is implemented in Pytorch.
 
 Prostate 3D dataset needs to be downloaded to this folder. see v1 folder. 
 
 Some other python packages that are required for running are listed below.
 
 [numpy](https://numpy.org/)
 
 [nibabel](https://nipy.org/nibabel/)
 
 [torchio](https://torchio.readthedocs.io/)
 
 [Pillow](https://pillow.readthedocs.io/en/stable/)
 
 In addition, considering the large size of the parameters in the model, it is recommended to run model.py on a GPU with more than 10GB vRAM. 
 
 ## Experiment preparing
 
  Prostate 3D data set is randomly split into 3 sets as the training set, the validation set, and the test set. The training set contains 179 samples. Both the validation set and the test set have 16 samples.
 
  As a data augmentation trick, the blur, spike and bias_field operations are randomly applied for loaded data in order to enlarge the size of the dataset.
 
  For training the model, an unweighted cross entropy loss function is used as the loss function. The optimizer uses Adam algorithm. The batch size is set to be 1.
  
 ## Experiment Result
  
  The experiment took 35 epochs and average Dice Similarity Coefficient of all 6 labels reached over 0.9. 
  
  ![image](https://github.com/aCoalBall/segmentation-of-3d-prostate/blob/main/performance.png)
  
  
  
  The following figure is a result on Case_004_Week0_LFOV.nii.gz.
  
  ![image](https://github.com/aCoalBall/segmentation-of-3d-prostate/blob/main/visualization/axial.png)
  ![image](https://github.com/aCoalBall/segmentation-of-3d-prostate/blob/main/visualization/coronal.png)
  ![image](https://github.com/aCoalBall/segmentation-of-3d-prostate/blob/main/visualization/sagittal.png)
  
  
  ## Reference
  
Çiçek, A. Abdulkadir, S. S. Lienkamp, T. Brox, and O. Ronneberger, “3D U-Net: Learning Dense
Volumetric Segmentation from Sparse Annotation,” in Medical Image Computing and Computer-Assisted Intervention – MICCAI 2016, ser. Lecture Notes in Computer Science, S. Ourselin, L. Joskowicz, M. R. Sabuncu,
G. Unal, and W. Wells, Eds. Cham: Springer International Publishing, 2016, pp. 424–432.

Paul A. Yushkevich, Joseph Piven, Heather Cody Hazlett, Rachel Gimpel Smith, Sean Ho, James C. Gee, and Guido Gerig. User-guided 3D active contour segmentation of anatomical structures: Significantly improved efficiency and reliability. Neuroimage. 2006 Jul 1; 31(3):1116-28. 
[bibtex] [medline] [doi:10.1016/j.neuroimage.2006.01.015]



  
  
  
  
  
  
  
  
  
  
