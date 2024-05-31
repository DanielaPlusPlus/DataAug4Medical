# LCAMix

Pytorch codes of following paper:


Danyang Sun., Fadi Dornaika., & Jinan Charafeddine. (2024). LCAMix: Local-and-contour aware grid mixing based data augmentation for medical image segmentation. Information Fusion, 102484.


https://doi.org/10.1016/j.inffus.2024.102484

In this work, we introduce LCAMix, a novel data augmentation approach designed for medical image segmentation. LCAMix operates by blending two images and their segmentation masks based on their superpixels, incorporating a local-and-contour-aware strategy. The training process on augmented images adopts two auxiliary pretext tasks: firstly, classifying local superpixels in augmented images using an adaptive focal margin, leveraging segmentation ground truth masks as prior knowledge; secondly, reconstructing the two source images using mixed superpixels as mutual masks, emphasizing spatial sensitivity. Our method stands out as a simple, one-stage, model-agnostic, and plug-and-play data augmentation solution applicable to various segmentation tasks. Notably, it requires no external data or additional models. Extensive experiments validate its superior performance across diverse medical segmentation datasets and tasks. 




Datasets:
=============================

ISIC datastes come from the [International Skin Imaging Collaboration challenge](https://challenge.isic-archive.com/data/). 

Synapse, GlaS and MoNuSeg datasets come from the [repo of TransUnet](https://github.com/Beckschen/TransUNet)

Thanks the authors of TransUnet for their sharing.

@article{chen2021transunet,
  title={TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation},
  author={Chen, Jieneng and Lu, Yongyi and Yu, Qihang and Luo, Xiangde and Adeli, Ehsan and Wang, Yan and Lu, Le and Yuille, Alan L., and Zhou, Yuyin},
  journal={arXiv preprint arXiv:2102.04306},
  year={2021}
}




