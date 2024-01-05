# PGNet
Position-Guided Infrared Small Target Detection
## Algorithm Introduction
we propose in this paper a robust infrared small target detection method jointing multiple information and noise prediction, named MINP-Net.


## Prerequisite
* Tested on Ubuntu 20.04 and 1x NVIDIA 2080Ti 
* [The SIRST download dir](https://github.com/YimianDai/sirst) 
* [The NUDT-SIRST download dir](https://github.com/YeRen123455/Infrared-Small-Target-Detection)

## Usage
#### 1. Train.

```bash
python train.py
```

#### 2. Test.

```bash
python test.py 
```

#### (Optional 1) Visulize your predicts.

```bash
python visulization.py
```

## Referrences
1. B. Li, C. Xiao, L. Wang, Y. Wang, Z. Lin, M. Li, W. An, and Y. Guo, “Dense Nested Attention Network for Infrared Small Target Detection,” IEEE Transactions on Image Processing, vol. 32, pp. 1745–1758, 2023. [[code]](https://github.com/YeRen123455/Infrared-Small-Target-Detection) 
