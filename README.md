# Human Parsing (Human body Part segmentation) using PyTorch

### Results:
<img src="results/00005_00.jpg" width="450"/> <img src="results/00005_00.png" width="450"/> 

Dataset

- This model is trained on [CCIHP](https://kalisteo.cea.fr/wp-content/uploads/2021/09/README.html) dataset which contains 22 class labels.

Please download imagenet pretrained resent-101 from [baidu drive](https://pan.baidu.com/s/1NoxI_JetjSVa7uqgVSKdPw) or [Google drive](https://drive.google.com/open?id=1rzLU-wK6rEorCNJfwrmIu5hY2wRMyKTK), and put it into dataset folder.

#### Training 

- Set necessary arguments and run `train_simplified.py`.

Citation:
```
@InProceedings{Liu_2022_CVPR,
    author    = {Liu, Kunliang and Choi, Ouk and Wang, Jianming and Hwang, Wonjun},
    title     = {CDGNet: Class Distribution Guided Network for Human Parsing},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2022},
    pages     = {4473-4482}
}
```

