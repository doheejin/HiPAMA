# HiPAMA 

This repository is the implementation of the paper, [**Hierarchical Pronunciation Assessment with Multi-Aspect Attention**](https://ieeexplore.ieee.org/document/10095733/) (ICASSP 2023).

> Our code is based on the open source, [https://github.com/YuanGongND/gopt](https://github.com/YuanGongND/gopt) (Gong et al, 2022).

## Citation
Please cite our paper if you find this repository helpful.

```
@INPROCEEDINGS{10095733,
  author={Do, Heejin and Kim, Yunsu and Lee, Gary Geunbae},
  booktitle={ICASSP 2023 - 2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={Hierarchical Pronunciation Assessment with Multi-Aspect Attention}, 
  year={2023},
  volume={},
  number={},
  pages={1-5},
  doi={10.1109/ICASSP49357.2023.10095733}}
```

## Dataset

An open source dataset, SpeechOcean762 (licenced with CC BY 4.0) is used. You can download it from [https://www.openslr.org/101](https://www.openslr.org/101).

## Package Requirements

Install below packages in your virtual environment before running the code.
- python version 3.8.10
- pytorch version '1.13.1+cu117'
- numpy version 1.20.3
- pandas version 1.5.0

You can run below command on your virtual environment
- `pip install -r requirements.txt`

## Training and Evaluation (HiPAMA)
This bash script will run each model 5 times with ([0, 1, 2, 3, 4]).
- `cd src`
- `bash run_hipama.sh`

Note that every run does not produce the same results due to the random elements.

The reported results in the paper are the averages of the final epoch results for five different seeds.

## Run baseline (GOPT)
This bash script will run each model 5 times with ([0, 1, 2, 3, 4]).
- `cd src`
- `bash run_gopt.sh`
