# KeepAugment_Pytorch


## Unofficial implementation of ["KeepAugment: A Simple Information-Preserving Data Augmentation Approach"](https://openaccess.thecvf.com/content/CVPR2021/papers/Gong_KeepAugment_A_Simple_Information-Preserving_Data_Augmentation_Approach_CVPR_2021_paper.pdf). CVPR2021

<p align="center"><img src="https://user-images.githubusercontent.com/53032349/171586571-6de784a5-0e34-4d63-9034-0fab7e62f69e.png" width="100%" height="100%" title="70px" alt="memoryblock"></p><br>

## Results 

### CIFAR 10

|              Model            |     ResNet-18     |  Wide ResNet-28-10  |     Shake-Shake     |
| :---------------------------: | :---------------: | :-----------------: | :-----------------: |
| Cutout                        | 95.6±0.1 (paper)  |  96.9±0.1 (paper)   |        TODO         |
| KeepCutout                    |        96.0       |        TODO         |        TODO         |
| KeepCutout (low resolution)   |        96.3       |        TODO         |        TODO         |
| KeepCutout (early loss)       |        96.2       |        TODO         |        TODO         |

|              Model              |     ResNet-18     |  Wide ResNet-28-10  |     Shake-Shake     |
| :-----------------------------: | :---------------: | :-----------------: | :-----------------: |
| AutoAugment                     |        TODO       |   97.3±0.1 (paper)  |   97.4±0.1 (paper)  |
| KeepAutoAugment                 |        TODO       |        TODO         |        TODO         |
| KeepAutoAugment (low resolution)|        TODO       |        TODO         |        TODO         |
| KeepAutoAugment (early loss)    |        TODO       |        TODO         |        TODO         |


## Run

The type of method is <br>
'keep_cutout', 'keep_cutout_low', 'keep_cutout_low_early'(TODO), 'keep_autoaugment', 'keep_autoaugment_low', 'keep_autoaugment_low_early'(TODO). 

The type of model is 'resnet', 'wide_resnet', 'shake'.

```Shell
python train.py --model resnet --method keep_cutout
```

## Referenced. Thank you all:+1:
baseline & cutout code : https://github.com/uoguelph-mlrg/Cutout<br>
randaugment code : https://github.com/ildoonet/pytorch-randaugment<br>
saliency map code : https://github.com/sunnynevarekar/pytorch-saliency-maps<br>
