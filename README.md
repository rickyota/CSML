# CSML    ver. 1.0.0
Cell Segmenter using Machine Learning.  
Software based on this code is distributed on [(Releases)](https://github.com/RickyOta/CSML/releases).  
Details are on [(Wiki)](https://github.com/RickyOta/CSML/wiki).

<img src="https://github.com/RickyOta/CSML/wiki/Images/example_infer_concat.png" height="256px">

## Description
CSML returns segmented images of fluorescent inference images, inferred by FCN classifier.  
Classifier is optimized by training and corresponding labeled images.  


## Usage of Software
1. Launch CSML.exe.
1. Train model (only for the first time).
	1. Drag & Drop the images file and the corresponding binary images file.  
	1. Specify name of model.
	1. Adjust the parameters if you want.
	1. Press "Start Training" button.
1. Infer images.
	1. Drag & Drop the images file which you want to segment. 
	1. Select the model which you want to use.
	1. Adjust the parameters if you want.
	1. Select output folder.
	1. Press "Start Infering" button.
1. Get segmented images file in output folder.

### Notes

- Click "Run with Example" to use examples.
- Format of Images can be a folder containing images, one tiff file or one other image file.  
- Accuracy displayed in model list is dot accuracy.


## Authors
[R. Ota](https://github.com/RickyOta)  
[R. Nakabayashi](https://github.com/ryought) (for help creating software)

---

##  Usage of Codes
Please refer to this section if you want to run the codes directly.

### Requirement
- Python >=3.6
- Chainer >=3.2
- OpenCV >=3.4


### Installation
```
git clone https://github.com/RickyOta/CSML.git
```


### Usage
1. Add images to ```CSML/data/```. The format can be a folder containing images, one tiff file or one other image file.
1. Edit values in ```CSML/train_infer_example.ini``` for training and inference or in ```CSML/infer_example.ini``` for only inference.  
	You can edit ini file name as long as it starts with 'train_infer' or 'infer'.
1. Execute
	```
	# for training classifier and inferring images 
	python src/main.py train_infer.ini
	# for only inferring images.
	python src/main.py infer.ini
	```
1. Inferred images are saved in ```CSML/result/```.

