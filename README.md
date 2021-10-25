![CI pipeline](https://github.com/mintusf/land_cover_tracking/actions/workflows/unittests.yml/badge.svg?branch=main)
![Flake8 check](https://github.com/mintusf/land_cover_tracking/actions/workflows/flake8.yml/badge.svg?branch=main)

## Summary
Land cover detection using Sentinel satellite data with instance segmentation.

## Folders and scripts structure:
* assets: images for README
* config: training configs, dataset configs
* data_analysis: EDA notebooks
* env: scripts to build docker environment
* logs: training logs
* models: models implementation
* tests: code tests
* tools: visualization tool
* train_utils: utils for training
* utils: I/O, visualization, raster utils
* weights: for weights
* train.py: used for training
* test.py: used for evaluating test set
* infer.py: used for inference (raster or alphablend)

## Dataset
Reference:
* https://github.com/chrieke/awesome-satellite-imagery-datasets/blob/master/README.md
* https://www.isprs-ann-photogramm-remote-sens-spatial-inf-sci.net/IV-2-W7/153/2019/isprs-annals-IV-2-W7-153-2019.pdf

### Used datasets:
SEN12MS ([LINK](https://mediatum.ub.tum.de/1474000))
* Spatial resolution: 10m
* Patch size: 256 x 256
* Classes: 33 (converted to 12)
* Channels: 12 bands of Sentinel L2C (Only RBG bands used for training and its results are shown in this document)
* Dataset for class `snow` was created by this repository creator using [EO Browser](https://apps.sentinel-hub.com/eo-browser/)


## Models
* DeepLab v3
* HRNet ([implementation](https://github.com/HRNet/HRNet-Semantic-Segmentation), [paper](https://arxiv.org/pdf/1904.04514.pdf))


## Results

### **Metrics (test set)**

| class | recall | precision  |      f1|
|-|-|-|-|
| shrublands    | 0.85 |  0.73  |0.79|
| urban         | 0.70 |  0.60  |0.65|
| agricultural  | 0.90 |  0.76  |0.82|
| barren        | 0.43 |  1.00  |0.60|
| snow          | 0.91 |  0.93  |0.92|
| water         | 0.92 |  0.97  |0.94|
| dense_forest  | 0.42 |  0.77  |0.54|
| open_forest   | 0.53 |  0.54  |0.54|
| wetlands      | 0.66 |  0.23  |0.23|
| grasslands    | 0.48 |  0.38  |0.42|


### **Confusion matrix (test set)**
![Confusion_matrix](assets/matrix.png)


### **Generated masks example**

![Colors](assets/colors.png)

![Test res](assets/test_res.jpg)

#### Real-life Web App
The model is used in [this repository](https://github.com/mintusf/land_cover_tracking) as a backend for land cover classification.
