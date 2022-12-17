# An Advanced Deep Learning Framework for Video-based Diagnosis of ASD
This repository contains the implementation for the paper:
Miaomiao Cai, Mingxing Li, Zhiwei Xiong, Pengju Zhao, Enyao Li and Jiulai Tang, ["An Advanced Deep Learning Framework for Video-based Diagnosis of ASD"](https://link.springer.com/chapter/10.1007/978-3-031-16440-8_42)

# Dataset
### To protect the rights and privacy of our research participants, please do not repost our dataset.
A total of 82 children (ASD: 57; TD: 25) finish the experiments with their parents at home, and they all consent to use their data for our research.
All diagnostic labels (ASD or TD) are provided by the ADOS-2, which have been done independently of our experiments. 
We provide the face features extracted by the Openface 2.0 toolbox.
![dataset]https://github.com/xiaotaiyangcmm/DASD/blob/main/figure/dataset.png

# Framework
![framework](https://github.com/xiaotaiyangcmm/DASD/blob/main/figure/pipeline.png)

# Prerequisites
- numpy            1.17.4 
- torch            1.3.1
- torchvision      0.4.2   

# commands
#### Training stage
- In `/code/main.py`, please change `rootpath`. 
- For 2 gpus:
```sh
python python /code/main.py --mynum 100 --numsegments 4 --model_name attention
```
#### Test stage
- In `/code/test.py`, please change `rootpath`. 
```sh
python python /code/test.py --mynum 100 --numsegments 4 --model_name attention
```


## Citing

If you use our code in your research or applications, please consider citing our paper.

```
@inproceedings{cai2022advanced,
  title={An Advanced Deep Learning Framework for Video-Based Diagnosis of ASD},
  author={Cai, Miaomiao and Li, Mingxing and Xiong, Zhiwei and Zhao, Pengju and Li, Enyao and Tang, Jiulai},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={434--444},
  year={2022},
  organization={Springer}
}
```
