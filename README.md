## [CVPR 2022] Optical Flow Estimation for Spiking Camera ([Paper](https://openaccess.thecvf.com/content/CVPR2022/html/Hu_Optical_Flow_Estimation_for_Spiking_Camera_CVPR_2022_paper.html))

<h4 align="center"> Liwen Hu<sup>1,2#</sup>, Rui Zhao<sup>1#</sup>, Ziluo Ding<sup>1</sup>, Lei Ma<sup>1,2</sup>, Boxin Shi<sup>1,2,3</sup>, Ruiqin Xiong<sup>1</sup>, Tiejun Huang<sup>1,2,3</sup> </h4>
<h4 align="center">1. NERCVT, School of Computer Science, Peking University <br>
    2. Beijing Academy of Artificial Intelligence <br>
	3. Institute for Artificial Intelligence, Peking University</h4><br> 


Spiking Camera Simulator (SPCS) sees https://github.com/Acnext/Spiking-Camera-Simulator-SPCS

## Environments

You will have to choose cudatoolkit version to match your compute environment. The code is tested on PyTorch 1.10.2+cu113 and spatial-correlation-sampler 0.3.0 but other versions might also work

```
conda create -n scflow python==3.9
conda activate scflow
conda install pytorch torchvision cudatoolkit=11.3 -c pytorch
pip3 install spatial-correlation-sampler opencv-python h5py tensorboardX
```

## Prepare the Data

```bash
# for training
bash encoding/proc_spift.sh
# for evaluating
bash encoding/proc_phm.sh
```

## Evaluate

```bash
# for dt=10 case
python3 eval.py -dt=10 --pretrained='./ckpt/dt10_e40.pth.tar'
# for dt=20 case
python3 eval.py -dt=20 --pretrained='./ckpt/dt20_e80.pth.tar'
```

## Train

```bash
# for dt=10 case
python3 train_scflow.py -dt=10 --epochs=40 -b=4 --lr=1e-4 \
--decay=0.7 --milestones 5 10 20 30
# for dt=20 case
python3 train_scflow.py -dt=20 --epochs=80 -b=4 --lr=1e-4 \
--decay=0.6 --milestones 5 15 25 35 45 55 65 75
```

## Datasets

In this work, two datasets are proposed where SPIkingly Flying Things (SPIFT) is as train set and Photo-realistic High-speed Motion (PHM) is as test set. You can download them from https://pan.baidu.com/s/1A5U9lsNyViGEQIyulSE8vg (password:5331).

### SPIFT

The dataset includes 100 scenes and each scene describes that different kinds of objects translate and rotate in random background includes spike streams with 500 frames, corresponding GT images and optical flow. Some scenes is shown in Fig. 1.
<img src="https://github.com/Acnext/Optical-Flow-For-Spiking-Camera/blob/main/figs/spift.jpg" width="100%">

### PHM

Each scene in the dataset is carefully designed and has a lot in common with the real world as shown in Fig. 2.  Besides, the number of generated spike frames for each scene is shown in Table. 1.

For PHM dataset, the motion distribution of "Fly" is extreme. The subsequent methods can use this scene optionally.

<img src="https://github.com/Acnext/Optical-Flow-For-Spiking-Camera/blob/main/figs/phm.jpg" width="100%">

| Ball | Cook | Dice | Doll | Fan  | Fly  | Hand | Jump | Poker | Top  |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ----- | ---- |
| 1000 | 4000 | 4000 | 3000 | 1000 | 4500 | 2000 | 1400 | 3200  | 1000 |

## Citation

If you think this work is helpful, please cite

```
@inproceedings{hu2022optical,
  title={Optical Flow Estimation for Spiking Camera},
  author={Hu, Liwen and Zhao, Rui and Ding, Ziluo and Ma, Lei and Shi, Boxin and Xiong, Ruiqin and Huang, Tiejun},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages={17844--17853},
  year={2022}
}
```