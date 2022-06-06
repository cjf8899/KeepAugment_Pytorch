# KeepAugment_Pytorch


## Unofficial implementation of ["KeepAugment: A Simple Information-Preserving Data Augmentation Approach"](https://openaccess.thecvf.com/content/CVPR2021/papers/Gong_KeepAugment_A_Simple_Information-Preserving_Data_Augmentation_Approach_CVPR_2021_paper.pdf). CVPR2021

<p align="center"><img src="https://user-images.githubusercontent.com/53032349/171586571-6de784a5-0e34-4d63-9034-0fab7e62f69e.png" width="100%" height="100%" title="70px" alt="memoryblock"></p><br>

## Results 

### CIFAR 10

|              Model            |     ResNet-18     |  Wide ResNet-28-10  |
| :---------------------------: | :---------------: | :-----------------: |
| Cutout                        |        95.6       |        96.9         |
| KeepCutout                    |        96.2       |        97.2         |
| KeepCutout (low resolution)   |        96.1       |        97.1         |
| KeepCutout (early loss)       |        96.2       |        TODO         |
| KeepCutout (low + early)      |        96.2       |        TODO         |

|              Model              |     ResNet-18     |  Wide ResNet-28-10  |
| :-----------------------------: | :---------------: | :-----------------: |
| AutoAugment                     |       96.0        |          97.2       |
| KeepAutoAugment                 |       96.4        |          97.4       |
| KeepAutoAugment (low resolution)|       96.3        |          97.5       |
| KeepAutoAugment (early loss)    |       96.3        |          97.4       |
| KeepAutoAugment (low + early)   |       96.5        |         TODO        |

All results have slight differences(±0.1).

## Run

The type of method is <br>
'cutout', 'keep_cutout', 'keep_cutout_low', 'keep_cutout_early', 'keep_cutout_low_early',<br>
'autoaugment', 'keep_autoaugment', 'keep_autoaugment_low', 'keep_autoaugment_early', 'keep_autoaugment_low_early'. 

The type of model is 'resnet', 'wide_resnet'.

```Shell
python train.py --model resnet --method keep_cutout
```

### Any feedback on code simplification and incorrect implementation would be appreciated!

## Referenced. Thank you all:+1:
baseline & cutout code : https://github.com/uoguelph-mlrg/Cutout<br>
autoaugment code : https://github.com/DeepVoltaire/AutoAugment<br>
saliency map code : https://github.com/sunnynevarekar/pytorch-saliency-maps<br>
