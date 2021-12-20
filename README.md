# Cross Sample Adversarial Debiasing

## Installation
Our code is developed based on the [public code](https://github.com/feidfoe/learning-not-to-learn) [Kim et al. CVPR19]. Thanks Kim! 

Requirements: Pytorch, torchvision, tqdm.

You need a GPU to run the code.

## Data Preparation
Download the colored mnist from [dataset](https://drive.google.com/file/d/11K-GmFD5cg3_KTtyBRkj9VBEnHl-hx_Q/view)

Extracted the x.npy files to ./colored_mnist/

## Experiments
We provide an example for coloredmnist with sigma^2=0.020, and one could change the parameters accordingly to conduct experiemnts on other settings.

We provide a pretrained baseline model for sigma^2=0.020. Conduct training by
```
 python main.py -e csad --color_var 0.020 --checkpoint baseline/checkpoint_step_0028.pth --alpha 1 --tau 10 --lambda_ 1 --save_dir ./
```
After training, you should get a debiased digit classifer with accuracy around 0.943. We fix the random seed for reproducibility and please erase checkpoint if you conduct experiments with other random seeds and settings.

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