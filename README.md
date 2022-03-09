# A Simple and Efficient Reconstruction Backbone for Snapshot Compressive Imaging


This repository contains the test code and pre-trained models for the [arXiv paper](https://arxiv.org/abs/2108.07739) by [Jiamian Wang](https://github.com/Jiamian-Wang), [Yulun Zhang](http://yulunzhang.com/), [Xin Yuan](https://www.bell-labs.com/about/researcher-profiles/xyuan/), [Yun Fu](http://www1.ece.neu.edu/~yunfu/) and [Zhiqiang Tao](http://ztao.cc/).

![framework](https://github.com/Jiamian-Wang/HSI_baseline/blob/main/framework.png) 

## Data

The benchmark simulation test dataset can be accessed [here](https://github.com/Jiamian-Wang/HSI_baseline/tree/main/Data/testing/simu). Make sure to specify the ```test_path``` before evaluation.
The coded apture can be accessed [here](https://github.com/Jiamian-Wang/HSI_baseline/tree/main/Data). Make sure to specify the ```mask_path``` before evaluation. 

## Three versions of proposed method

Checkpoint of ```model v1```: w/o rescaling pairs can be accessed [here](https://github.com/Jiamian-Wang/HSI_baseline/tree/main/models/v1)
Similarly, pre-trained models of the other two versions are also been released. 
Makre sure to specify the ```model_path``` as desired pre-trained model. 

## Citation

If you find new baseline useful, please cite the following paper

``` 
@article{wang2021new,
  title={A new backbone for hyperspectral image reconstruction},
  author={Wang, Jiamian and Zhang, Yulun and Yuan, Xin and Fu, Yun and Tao, Zhiqiang},
  journal={arXiv preprint arXiv:2108.07739},
  year={2021}
}
```
For any questions, please submit an issue or contact [jiamiansc@gamil.com](jiamiansc@gamil.com). 
