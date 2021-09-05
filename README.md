![CI pipeline](https://github.com/mintusf/land_cover_tracking/actions/workflows/unittests.yml/badge.svg?branch=main)
![Flake8 check](https://github.com/mintusf/land_cover_tracking/actions/workflows/flake8.yml/badge.svg?branch=main)

## Summary
Land cover detection using Sentinel satellite data with instance segmentation (WIP).

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
* Input channels: "B2", "B3", "B4", "B5" of Sentinel L2C

### Other datasets:
* only water: [LINK](https://www.kaggle.com/franciscoescobar/satellite-images-of-water-bodies)
* surface & cloud: [LINK](https://zenodo.org/record/4172871#.YQYu_44zZPY)
* surface (only Africa): [LINK](https://registry.mlhub.earth/10.34911/rdnt.d2ce8i/)
* only Slovenia: [LINK](http://eo-learn.sentinel-hub.com/)
* big Europe: [LINK](http://bigearth.net/#about)

## Models
* Model: DeepLab v3


## Results

**Confusion matrix:**

![Confusion_matrix](assets/Confusion_matrix.PNG)


**Generated masks example**

| urban | water | agriculture | wetlands | open_forest | deep_forest |
|-|-|-|-|-|-|
| ![urban](assets/urban.PNG) | ![water](assets/water.PNG) | ![agriculture](assets/agriculture.PNG) | ![wetlands](assets/wetlands.PNG) | ![open_forest](assets/open_forest.PNG) | ![deep_forest](assets/deep_forest.PNG) |

![Results1](assets/results1.PNG)
![Results2](assets/results2.PNG)
![Results3](assets/results3.PNG)
![Results4](assets/results4.PNG)