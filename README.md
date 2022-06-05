# KeepAugment_Pytorch


## Unofficial implementation of ["KeepAugment: A Simple Information-Preserving Data Augmentation Approach"](https://openaccess.thecvf.com/content/CVPR2021/papers/Gong_KeepAugment_A_Simple_Information-Preserving_Data_Augmentation_Approach_CVPR_2021_paper.pdf). CVPR2021

<p align="center"><img src="https://user-images.githubusercontent.com/53032349/171586571-6de784a5-0e34-4d63-9034-0fab7e62f69e.png" width="100%" height="100%" title="70px" alt="memoryblock"></p><br>

## Results 

### CIFAR 10

|              Model            |     ResNet-18     |  Wide ResNet-28-10  |
| :---------------------------: | :---------------: | :-----------------: |
| Cutout                        |        0.956      |      0.9691         |
| KeepCutout                    |        0.962      |      0.9721         |
| KeepCutout (low resolution)   |        0.961      |      0.9719         |
| KeepCutout (early loss)       |        0.9621     |        TODO         |
| KeepCutout (low + early)      |        TODO       |        TODO         |

|              Model              |     ResNet-18     |  Wide ResNet-28-10  |
| :-----------------------------: | :---------------: | :-----------------: |
| AutoAugment                     |       0.9607      |        0.9722       |
| KeepAutoAugment                 |       0.9646      |        0.9747       |
| KeepAutoAugment (low resolution)|       0.9639      |        0.9753       |
| KeepAutoAugment (early loss)    |       0.9635      |        0.9748       |
| KeepAutoAugment (low + early)   |       TODO        |         TODO        |


## Run

The type of method is <br>
'cutout', 'keep_cutout', 'keep_cutout_low', 'keep_cutout_early', 'keep_cutout_low_early',<br>
'autoaugment', 'keep_autoaugment', 'keep_autoaugment_low', 'keep_autoaugment_early', 'keep_autoaugment_low_early'. 

The type of model is 'resnet', 'wide_resnet'.

```Shell
python train.py --model resnet --method keep_cutout
```

## Referenced. Thank you all:+1:
baseline & cutout code : https://github.com/uoguelph-mlrg/Cutout<br>
autoaugment code : https://github.com/DeepVoltaire/AutoAugment<br>
saliency map code : https://github.com/sunnynevarekar/pytorch-saliency-maps<br>
