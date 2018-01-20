# CSML    ver. 1.0.0
Cell Segmenter using Machine Learning.  
Software based on this code is distributed [(here)](https://github.com/RickyOta/CSML/releases).  


## Description
Return segmented images of fluorescent ones, inferred by FCN classifier optimized by training images.  


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
[R. Ota](https://github.com/RickyOta)




---

##  Usage of Codes
Please refer to this section, if you want to run the codes directly.

### Requirement
- Python >=3.6
- Chainer >=3.2
- OpenCV >=3.4


### Installation
```
git clone https://github.com/RickyOta/CSML.git
```


### Usage
1. Add images to ```CSML/data/```. The format can be folder containing images or one tiff file.
1. Edit values in ```CSML/train_infer.ini``` for training and inference or ```CSML/infer.ini``` for only inference.  
1. Execute
	```
	python src/main.py train_infer.ini
	```
	or
	```
	python src/main.py infer.ini
	```	
1. Inferred images are saved in ```CSML/result/```.

### Details
#### Format of images
if you specify folder as output, inferred images are saved in the folder.
