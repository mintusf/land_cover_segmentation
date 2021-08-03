![CI pipeline](https://github.com/mintusf/land_cover_tracking/actions/workflows/unittests.yml/badge.svg?branch=main)
![Flake8 check](https://github.com/mintusf/land_cover_tracking/actions/workflows/flake8.yml/badge.svg?branch=main)

### Summary
Using Sentinel-2 data to detect percentage of various land types (WIP).

### Dataset
Used for searching:
* https://github.com/chrieke/awesome-satellite-imagery-datasets/blob/master/README.md
* https://www.isprs-ann-photogramm-remote-sens-spatial-inf-sci.net/IV-2-W7/153/2019/isprs-annals-IV-2-W7-153-2019.pdf

Used datasets:
* only water: [LINK](https://www.kaggle.com/franciscoescobar/satellite-images-of-water-bodies) DONE
* surface & cloud: [LINK](https://zenodo.org/record/4172871#.YQYu_44zZPY)
* surface (only Africa): [LINK](https://registry.mlhub.earth/10.34911/rdnt.d2ce8i/)
* only Slovenia: [LINK](http://eo-learn.sentinel-hub.com/)
* big Europe: [LINK](http://bigearth.net/#about)
* whole world: [LINK](https://mediatum.ub.tum.de/1474000)


| Name | Sensor | GR (m) | Geographical area | Dataset size (GB) | Subgrid size (pxl x pxl) | Classes count |
|-|-|-|-|-|-|-|
| Satellite-images-of-water-bodies | Sentinel-2 | ? | ? | 0.25 | various (up to 3000x3000)  | 1 [water] |
| Sentinel-2 Cloud Mask Catalogue | 2018 Level-1C Sentinel-2 | 20 |  | 15.3 (half without clouds) | 1022 x 1022 | 11 [est/jungle, snow/ice, agricultural, urban/developed, coastal, hills/mountains, desert/barren, shrublands/plains, wetland/bog/marsh, open_water, enclosed_water] + clouds |
| LandCoverNet | Sentinel-2 L2A | 10 | Africa | 82 | 256 x 256 | 7 [water, artificial bare ground, natural bare ground, snow/ice, woody vegetation, cultivated vegetation, semi natural vegetation] |
| [Example dataset of EOPatches for Slovenia 2019](http://eo-learn.sentinel-hub.com/) | Sentinel-2 L1C | 10 | Slovenia | 11 | 1000 x 1000 | 9 [cultivated land, forest, grassland, shrubland, water, wetlands, tundra, artificial surface, bareland] |
| BigEarthNet | Sentinel-2 L2A | 10 | Europe | 120 x 120 | CLASSIFICATION ! |
| SEN12MS | Sentinel-2 L2A | 10 | Whole world | 510 | 256 x 256 | 33 |