# CSML   ver.1.0.0
Cell Segmenter by Machine Learning.  
Software based on this code is distributed [(here)](http://...).  


## Description
Return segmented images of fluorescent images, inferred by FCN classifier optimized by training images.  


## Usage of Software
1. Launch CSML.exe.
1. Train model (for the first time).
	1. Drag & Drop the images file and the corresponding binary images file. Tiff files can be used for two or more images.
	1. Adjust the parameters if you want.
	1. Press "Train" button.
1. Infer images.
	1. Drag & Drop the images file which you want to segment.
	1. Select output folder.
	1. Adjust the parameters if you want.
	1. Press "Infer" button.
1. Get segmented images file in output folder.


## Publication


## Author
[R, Ota](https://github.com/RickyOta)



---

##  Usage of Codes
Please refer to this section, if you want to run codes directly.

### Requirement
- Python <3.6
- Chainer >=1.21
- OpenCV >=3.1


### Installation
```
git clone https://github.com/RickyOta/CSML.git
```


### Usage
1. Add images to ```CSML/data``` and edit paths of images in ```CSML/main.py```.  
1. Move into ```CSML/src``` and execute
```
python main.py
```
1. Inferred images are saved in ```FCN/result```.




