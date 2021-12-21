# Cross Sample Adversarial Debiasing

## Installation
Our code is developed based on the [public code](https://github.com/feidfoe/learning-not-to-learn) [Kim et al. CVPR19]. Thanks Kim! 

Requirements: Pytorch, torchvision, tqdm.

You need a GPU to run the code.

## Data Preparation
Download the colored mnist from [dataset](https://drive.google.com/file/d/11K-GmFD5cg3_KTtyBRkj9VBEnHl-hx_Q/view)

Extracted the x.npy files to ./colored_mnist/

## Experiments

We provide a pretrained baseline model for sigma^2=0.020, and one can conduct training for CSAD by
```
 python main.py -e csad0020 --color_var 0.020 --checkpoint baseline/pretraincheckpoint_step_0000.pth --alpha 1 --tau 10 --lambda_ 1 --save_dir ./
```
After training, you should get a debiased digit classifer with accuracy around 0.943. 

We fix the random seed for reproducibility. Please erase the checkpoint and change random seeds for experiments with differenet settings. 

Please check our paper for details, and thanks for your interests.

## Citation
```
@InProceedings{Zhu_2021_ICCV,
    author    = {Zhu, Wei and Zheng, Haitian and Liao, Haofu and Li, Weijian and Luo, Jiebo},
    title     = {Learning Bias-Invariant Representation by Cross-Sample Mutual Information Minimization},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2021},
    pages     = {15002-15012}
}
```