# PGNet

## Algorithm Introduction
we propose in this paper Position-Guided Infrared Small Target Detection, named PGNet.

## Prerequisite
* Tested on Ubuntu 20.04 and NVIDIA 3090 Ti 
* [The SIRST download dir](https://github.com/YimianDai/sirst) 
* [The NUDT-SIRST download dir](https://github.com/YeRen123455/Infrared-Small-Target-Detection)
* [The IRSTD-1k download dir](https://github.com/RuiZhang97/ISNet)

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
