# KeepAugment_Pytorch


## Unofficial implementation of ["KeepAugment: A Simple Information-Preserving Data Augmentation Approach"](https://openaccess.thecvf.com/content/CVPR2021/papers/Gong_KeepAugment_A_Simple_Information-Preserving_Data_Augmentation_Approach_CVPR_2021_paper.pdf). CVPR2021

<p align="center"><img src="https://user-images.githubusercontent.com/53032349/171586571-6de784a5-0e34-4d63-9034-0fab7e62f69e.png" width="100%" height="100%" title="70px" alt="memoryblock"></p><br>

## Results 

### CIFAR 10

|              Model            |     ResNet-18     |    ResNet-110     |  Wide ResNet-28-10  |
| :---------------------------: | :---------------: | :---------------: | :-----------------: |
| Cutout                        | 95.6±0.1 (paper)  | 94.8±0.1 (paper)  |  96.9±0.1 (paper)   |
| KeepCutout                    |        TODO       |        TODO       |        TODO         |
| KeepCutout (low resolution)   |        TODO       |        TODO       |        TODO         |
| KeepCutout (early loss)       |        TODO       |        TODO       |        TODO         |

|              Model              |     ResNet-18     |    ResNet-110     |  Wide ResNet-28-10  |
| :-----------------------------: | :---------------: | :---------------: | :-----------------: |
| AutoAugment                     |        95.3       |        TODO       |   97.3±0.1 (paper)  |
| KeepAutoAugment                 |        TODO       |        TODO       |        TODO         |
| KeepAutoAugment (low resolution)|        TODO       |        TODO       |        TODO         |
| KeepAutoAugment (early loss)    |        TODO       |        TODO       |        TODO         |


## Run

The type of method is 'keep_cutout', 'keep_cutout_low', 'keep_cutout_low_early'(TODO), 'keep_autoaugment', 'keep_autoaugment_low', 'keep_autoaugment_low_early'(TODO). 

The type of model is 'resnet18', 'wide_resnet_28_10'.

ex)
```Shell
python main.py --exps_name cifar10_keepcutout --method keep_cutout --model resnet18
```

## Referenced. Thank you all:+1:
baseline code : https://github.com/kuangliu/pytorch-cifar<br>
cutout code : https://github.com/uoguelph-mlrg/Cutout<br>
Randaugment code : https://github.com/ildoonet/pytorch-randaugment<br>
saliency map code : https://github.com/sunnynevarekar/pytorch-saliency-maps<br>
