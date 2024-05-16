# 3D Human Pose Estimation with Spatial and Temporal Transformers
This repo is the official implementation for [CrossFormer: Cross Spatio-Temporal Transformer for 3D Human Pose Estimation](https://arxiv.org/abs/2203.13387)  




Our code is built on top of [VideoPose3D](https://github.com/facebookresearch/VideoPose3D).

### Environment

The code is developed and tested under the following environment

* Python 3.8.2
* PyTorch 1.7.1
* CUDA 11.0

You can create the environment:
```bash
conda env create -f crossformer.yml
```

### Dataset

Our code is compatible with the dataset setup introduced by [Martinez et al.](https://github.com/una-dinosauria/3d-pose-baseline) and [Pavllo et al.](https://github.com/facebookresearch/VideoPose3D). Please refer to [VideoPose3D](https://github.com/facebookresearch/VideoPose3D) to set up the Human3.6M dataset  (./data directory). 

### Evaluating pre-trained models

We provide the pre-trained 81-frame model (CPN detected 2D pose as input) [here](https://drive.google.com/file/d/1eNmpfTAhc-6hKLQjXv7qeCdpVvZlI8D1/view?usp=sharing). To evaluate it, put it into the `./checkpoint` directory and run:

```bash
python run_crossformer.py -k cpn_ft_h36m_dbb -f 81 -c checkpoint --evaluate best_epoch44.4.bin
```

We also provide pre-trained 81-frame model (Ground truth 2D pose as input) [here](https://drive.google.com/file/d/1LF-HVcyqMWC8VBWDXrL4oVfGzGnpT8aN/view?usp=sharing). To evaluate it, put it into the `./checkpoint` directory and run:

```bash
python run_crossformer.py -k gt -f 81 -c checkpoint --evaluate best_epoch_gt_28.5.bin
```


### Training new models

* To train a model from scratch (CPN detected 2D pose as input), run:

```bash
python run_crossformer.py -k cpn_ft_h36m_dbb -f 27 -lr 0.00004 -lrd 0.99
```

 

* To train a model from scratch (Ground truth 2D pose as input), run:

```bash
python run_crossformer.py -k gt -f 81 -lr 0.0004 -lrd 0.99
```

81 frames achieves 28.5 mm (MPJPE). 

### Visualization and other functions

We keep our code consistent with [VideoPose3D](https://github.com/facebookresearch/VideoPose3D). Please refer to their project page for further information. 

## Acknowledgement

Part of our code is borrowed from [VideoPose3D](https://github.com/facebookresearch/VideoPose3D). We thank the authors for releasing the codes.
