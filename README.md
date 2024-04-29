# Effective Semi-Supervised Medical Image Segmentation with Probabilistic Representations and Prototype Learning
by Yuchen Yuan, Xi Wang, Xikai Yang, Ruijiang Li, and Pheng-Ann Heng
## Introduction
Official code for "Effective Semi-Supervised Medical Image
Segmentation with Probabilistic Representations
and Prototype Learning".
## Requirements
This repository is based on PyTorch 1.13.0, CUDA 11.7 and Python 3.7.13. All experiments in our paper were conducted on NVIDIA GeForce RTX 4090 GPU with an identical experimental setting.
## Usage
We provide `code`, `data_split` for LA and ACDC dataset.

Data could be got at [LA](https://github.com/yulequan/UA-MT/tree/master/data) and [ACDC](https://github.com/HiLab-git/SSL4MIS/tree/master/data/ACDC).

To train a model,
```
python ./code/LA_train.py  #for LA training
python ./code/ACDC_train.py  #for ACDC training
``` 

To test a model,
```
python ./code/test_LA.py  #for LA testing
python ./code/test_ACDC.py  #for ACDC testing
```

[comment]: <> (## Citation)

[comment]: <> (If you find these projects useful, please consider citing:)

[comment]: <> (```bibtex)

[comment]: <> (@article{DBLP:journals/corr/abs-2305-00673,)

[comment]: <> (  author       = {Yunhao Bai and)

[comment]: <> (                  Duowen Chen and)

[comment]: <> (                  Qingli Li and)

[comment]: <> (                  Wei Shen and)

[comment]: <> (                  Yan Wang},)

[comment]: <> (  title        = {Bidirectional Copy-Paste for Semi-Supervised Medical Image Segmentation},)

[comment]: <> (  journal      = {CoRR},)

[comment]: <> (  volume       = {abs/2305.00673},)

[comment]: <> (  year         = {2023})

[comment]: <> (})

[comment]: <> (```)

## Acknowledgements
Our code is largely based on our prior work [SSCI](https://github.com/IsYuchenYuan/SSCI) and the work [PRCL](https://github.com/Haoyu-Xie/PRCL). 
## Questions
If you have any questions, welcome contact me at 'ycyuan22@ce.cuhk.edu.hk'



