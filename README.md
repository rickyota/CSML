# CSML    ver. 1.2.1
Cell Segmenter using Machine Learning.  
[Releases](https://github.com/rickyota/CSML/releases): Software based on this code.  
[Wiki](https://github.com/rickyota/CSML/wiki): Details of CSML.  

<img src="https://github.com/rickyota/CSML/wiki/image/example_infer_concat.png" height="256px">

<img src="https://github.com/rickyota/CSML/wiki/image/stats.png" width="512px">


## Description
CSML returns segmented images of fluorescent images, inferred by FCN classifier, tagged images and statistics of each cell.  
Classifier is optimized by training and corresponding labeled images.  
Statistics contain Centroids, Area, Perimeter, Eccentricity, Major axis, Minor axis, Orientation and Solidity.

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
- Format of Images is a folder containing images.  
- Accuracy displayed in model list is dot accuracy.


## Authors
[R. Ota](https://github.com/rickyota)  
[R. Nakabayashi](https://github.com/ryought) (for help creating software)

## Publication  
Ota R, Ide T, Michiue T. A rapid segmentation method of cell boundary for developing embryos using machine learning with a personal computer. Dev Growth Differ. 2021 Oct;63(8):406-416. doi: 10.1111/dgd.12747. Epub 2021 Sep 20. PMID: 34453320.

A novel cell segmentation method for developing embryos using machine learning  
Rikifumi Ota, Takahiro Ide, Tatsuo Michiue  
bioRxiv 288720; doi: https://doi.org/10.1101/288720

---

##  Usage of Code
Please refer to this section if you want to run the codes directly.

### Requirement
- conda 

### Installation
```
$ git clone https://github.com/rickyota/CSML.git
```

You need to use `conda` to install python dependencies.
```
$ conda env create --file environment.yml
```


### Usage
1. Add images to folders in ```./data/```.
    - Make sure that you set the same image name in `label` and `train` folder.
1. Execute
	```
    $ conda activate csml
	$ bash csml.sh
	```
1. Inferred images are saved in ```./result/```.


If you want to run inferring step only or run csml with various parameters, refer to `csml_onlyinfer.sh` and `csml_paras.sh`.

