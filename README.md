# PGNet

## 📁 Data Preparation

### 1\. Download the Datasets

First, please download the required datasets from the following links:

  * **SIRST**: [Download Link](https://github.com/YimianDai/sirst)
  * **NUDT-SIRST**: [Download Link](https://github.com/YeRen123455/Infrared-Small-Target-Detection)
  * **IRSTD-1k**: [Download Link](https://github.com/RuiZhang97/ISNet)

### 2\. Organize the Directory Structure

After downloading and extracting the files, place them in the `datasets` folder located in the project's root directory. Your project should adhere to the following structure:

```
PGNet/
├── datasets/
│   ├── IRSTD-1K/
│   │   ├── images/
│   │   │   ├── XDU0.png
│   │   │   └── ...
│   │   ├── masks/
│   │   │   ├── XDU0.png
│   │   │   └── ...
│   │   ├── train.txt
│   │   └── test.txt
│   │
│   ├── NUDT-SIRST/
│   │   ├── images/
│   │   │   ├── 000001.png
│   │   │   └── ...
│   │   ├── masks/
│   │   │   ├── 000001.png
│   │   │   └── ...
│   │   ├── train.txt
│   │   └── test.txt
│   │
│   └── SIRST/
│       ├── images/
│       │   ├── Misc_1.png
│       │   └── ...
│       ├── masks/
│       │   ├── Misc_1.png
│       │   └── ...
│       ├── train.txt
│       └── test.txt
│
├── train.py
└── test.py
```

-----

## 💡 Usage

### 1\. Training

Run the following command to start training the model. The script will automatically load the data from the `datasets` directory.

```bash
python train.py
```

We also provide pretrained model weights, which can be downloaded from the link below:

  * **Pretrained Models**: [Baidu Pan Link](https://pan.baidu.com/s/1vZeVvibTKP5zQawhAgN5FQ) (Password: `dq4b`)

### 2\. Testing

Use the `test.py` script to evaluate the model's performance. Please ensure you have either downloaded the pretrained models or completed the training process.

```bash
python test.py
```

For your convenience, we have also provided the prediction maps for the three datasets:

  * **Prediction Maps**: [Baidu Pan Link](https://pan.baidu.com/s/1jVfyBWyBvKZjgPB390aD4Q) (Password: `iigd`)

-----

## 🙏 Acknowledgements

The codebase for this project is heavily borrowed from the [IRSTD-Toolbox](https://github.com/XinyiYing/BasicIRSTD). We would like to express our sincere gratitude to the original author, **Xinyi Ying**, for their excellent work.