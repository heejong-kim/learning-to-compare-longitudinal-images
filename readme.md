# Learning to compare longitudinal images
Implementation of the paper "Learning to compare longitudinal images" of Heejong Kim and Mert Sabuncu, to appear in MIDL 2023 (Oral).

[Project Page](https://heejongkim.com/pairnet-midl) | [Paper](https://openreview.net/forum?id=l17YFzXLP53)

[comment]: <> (![Pairwise Image Ranking Network &#40;PaIRNet&#41;]&#40;figure-architecture.png&#41;)
![Example result of synthetic tumor](video-example.gif)


[comment]: <> (TODO: update figure to a video of tumor size detection)

## Dependencies 
```shell
conda env create -f environment.yml
conda activate pairnet
```


## Instructions
We provide an example script and dataset of synthetic tumor. The script includes training and experiments (weighted CAM and correlation).
```shell script
bash example-script-tumor.sh
```


## Citation
If you use this code, please consider citing our work:
```
@inproceedings{
kim2023learning,
title={Learning to Compare Longitudinal Images},
author={Heejong Kim and Mert R. Sabuncu},
booktitle={Medical Imaging with Deep Learning},
year={2023},
url={https://openreview.net/forum?id=l17YFzXLP53}
}```
