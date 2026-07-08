![Python >=3.11.11](https://img.shields.io/badge/Python->=3.11.11-yellow.svg)
![PyTorch >=2.2.2](https://img.shields.io/badge/PyTorch->=2.2.2-blue.svg)

# SAT-Mamba: State Association and Adaptive Topology Learning for Lifelong Ship Re-Identification
The *official* repository for [SAT-Mamba: State Association and Adaptive Topology Learning for Lifelong Ship Re-Identification]


## Requirements

### Installation
```bash
conda env create -f environment.yml
conda activate your_environment_name
```
We recommend to use one 24G RTX 3090 for training and evaluation. If you find some packages are missing, please install them manually. 


### Prepare Datasets

All the datasets should be downloaded on your own. All the datasets should be organized as follows:
```
dataroot
├── Market1501
│   └── bounding_box_train
│   └── bounding_box_test
│   └── query
├── MSMT17
│   └── bounding_box_train
│   └── bounding_box_test
│   └── query
├── ...
```

Please modify 'args.dataroot' to the corresponding path.

## Pre-trained Models 
|      Model      | Download |
|:---------------:| :------: |
|    MambaR-Small | [link](https://huggingface.co/Wangf3014/Mamba-Reg/resolve/main/mambar_small_patch16_224.pth) |

Please modify the pre-trained model path of '_backbones' in the 'model/reidmamba.py'.
Change '/root/data/.cache/models/mambar_small_patch16_224.pth' to your own pre-trained model path.


## Training

### Order 1: CMshipReID → VesselReID

```bash
python continual_main.py \
  --gpus 0 \
  --exp sat_mamba_order1 \
  --dataroot your/path/dataroot \
  --dataset CMshipReID,VesslReID \
  --unseen_dataset Warships-ReID,Sub-MARVEL
```
### Order 1: VesselReID → CMshipReID
```bash
python continual_main.py \
  --gpus 0 \
  --exp sat_mamba_order2 \
  --dataroot your/path/dataroot \
  --dataset VesslReID,CMshipReID \
  --unseen_dataset Warships-ReID,Sub-MARVEL
```



