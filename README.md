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
	1. Press "Start Training" button.
1. Infer images.
	1. Drag & Drop the images file which you want to segment. 
	1. Select the model which you want to use.
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
1. Add images to a file in ```CSML/data/```.
1. Execute  
	```
	# for training classifier and inferring images
	python src/main.py -t "example_train" -l "example_label" -o "example_result" -m "model_example.pkl"
	
	# for only inferring images.
	python src/main.py -f -i "example_infer" -o "example_result" -m "model_example.pkl"
	```  
	Details of all options are available ```python src/main.py --help```.
1. Inferred images are saved in ```CSML/result/```.

