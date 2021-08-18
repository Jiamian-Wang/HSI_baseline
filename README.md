# A New Backbone for Hyperspectral Image Reconstruction

This repository contains the test code and pre-trained models for the [arXiv paper](https://arxiv.org/abs/2108.07739).

![framework](https://github.com/Jiamian-Wang/HSI_baseline/blob/main/framework_v4.png) 

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
@misc{wang2021new,
      title={A New Backbone for Hyperspectral Image Reconstruction}, 
      author={Jiamian Wang and Yulun Zhang and Xin Yuan and Yun Fu and Zhiqiang Tao},
      year={2021},
      eprint={2108.07739},
      archivePrefix={arXiv},
      primaryClass={eess.IV}
}
```
For any questions, please submit an issue or contact [jiamiansc@gamil.com](jiamiansc@gamil.com). 
