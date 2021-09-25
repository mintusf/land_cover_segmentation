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

### **Metrics (test set)**

| class | recall | precision  |      f1|
|-|-|-|-|
| shrublands    | 0.931397 |  0.398179  |0.557866|
| urban         | 0.748475 |  0.679214  |0.712164|
| agricultural  | 0.883135 |  0.235822  |0.372244|
| barren        | 0.389128 |  0.851968  |0.534245|
| snow          | 1.000000 |  0.999835  |0.999917|
| water         | 0.818348 |  0.959264  |0.883221|
| dense_forest  | 0.659396 |  0.670900  |0.665098|
| open_forest   | 0.373307 |  0.728801  |0.493720|
| wetlands      | 0.473258 |  0.213453  |0.294209|
| grasslands    | 0.003759 |  0.612439  |0.007472|



![Confusion_matrix](assets/Confusion_matrix.PNG)


### **Generated masks example**

| urban | water | agriculture | wetlands | open_forest | deep_forest |
|-|-|-|-|-|-|
| ![urban](assets/urban.PNG) | ![water](assets/water.PNG) | ![agriculture](assets/agriculture.PNG) | ![wetlands](assets/wetlands.PNG) | ![open_forest](assets/open_forest.PNG) | ![deep_forest](assets/deep_forest.PNG) |

#### Test dataset

| Stree Map | Satelite image alphablended with predictions masks |
|-|-|
| ![1-1](assets/1-1.PNG) | ![1-2](assets/1-2.PNG) |
| ![2-1](assets/2-1.PNG) | ![2-2](assets/2-2.PNG) |
| ![3-1](assets/3-1.PNG) | ![3-2](assets/3-2.PNG) |
| ![4-1](assets/4-1.PNG) | ![4-2](assets/4-2.PNG) |
| ![5-1](assets/5-1.PNG) | ![5-2](assets/5-2.PNG) |
| ![6-1](assets/6-1.PNG) | ![6-2](assets/6-2.PNG) |
| ![7-1](assets/7-1.PNG) | ![7-2](assets/7-2.PNG) |

#### Real-life test (Sentinel image downloaded from EOBrowser)

![Results5](assets/Sentinel_data_1.PNG)