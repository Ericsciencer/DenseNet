# DenseNet (Densely Connected Convolutional Networks)
### 选择语言 | Language
[中文简介](#简介) | [English](#Introduction)

### 结果 | Result


---

## 简介
DenseNet（密集连接卷积网络）由黄高、Zhuang Liu等人于2017年提出，经典成果发表于《Densely Connected Convolutional Networks》，荣获CVPR 2017最佳论文奖。该网络在ResNet残差学习的基础上进一步升级，创新性提出**密集块（Dense Block）**与**过渡层（Transition Layer）**核心结构，以**跨层密集拼接**替代残差逐元素相加，构建层与层之间的全连接特征传递通路。通过极致的特征复用与多路梯度回流机制，彻底优化深层网络梯度消失问题，同时大幅压缩参数量与计算量，提升特征利用率与模型泛化能力。DenseNet凭借轻量化、强特征融合、易训练的优势，在ImageNet分类、下游视觉任务中取得优异效果，与ResNet共同成为深度卷积神经网络两大标杆骨架，广泛应用于目标检测、语义分割、医学影像、轻量化模型设计等领域。

## 架构
DenseNet核心架构为**密集块与过渡层交替堆叠**的分段式深度卷积网络，整体由**初始卷积层**、**多组串联密集模块**、**过渡下采样层**、**全局平均池化**与**全连接分类头**组成，彻底打破传统CNN单向串行的特征传递模式。原论文标准输入为224×224分辨率3通道RGB图像，原生支持DenseNet121/169/201/264等多深度版本，具体核心结构与设计如下：
- **核心特征单元（密集块 Dense Block）**：作为网络基础模块，块内每一层卷积都会接收前方所有层的特征图作为输入，采用通道拼接方式融合特征；通过固定增长率（Growth Rate）限制单层输出通道数，避免维度爆炸，以低参数量实现密集特征堆叠。
- **密集连接机制（Dense Connection）**：区别于ResNet短路相加的融合方式，DenseNet采用Concat通道拼接实现特征复用，每一层直接与后续所有层建立连接，梯度可通过多条路径反向传播，最大化保留梯度信息流，从结构层面强化深层网络可训练性。
- **过渡层与多尺度层级设计**：相邻密集块之间配置过渡层，依靠1×1卷积降维压缩通道、结合平均池化完成下采样，控制模型复杂度并缩小特征尺寸；网络逐级提取浅层边缘纹理、中层细节特征、高层全局语义，末端全局平均池化精简参数，有效抑制小数据集过拟合。

<img width="735" height="55" alt="image" src="https://github.com/user-attachments/assets/3815ad70-a20e-4f68-b9b8-5091e9e49bc7" />


该架构以「特征复用+密集融合」为核心创新，用更少参数实现更强特征表达能力，结构紧凑高效、梯度传播稳定，**密集连接**设计思想极大拓展了卷积网络的特征融合思路，是计算机视觉领域继残差网络后又一里程碑式创新。

<img width="784" height="633" alt="image" src="https://github.com/user-attachments/assets/49752de2-e811-4aa5-81be-4e3adec6c163" />
<img width="1205" height="265" alt="image" src="https://github.com/user-attachments/assets/98a80b07-72a6-4b70-ac79-878ad2376856" />
<img width="1199" height="590" alt="image" src="https://github.com/user-attachments/assets/fd4b9e9c-1773-4543-99f4-6d0fb8f952e8" />



**注意**：本文实验基于CIFAR-10十分类数据集开展，不同于原论文的ImageNet大规模数据集。因CIFAR-10图像尺寸仅为32×32，远小于原版224×224输入，因此对原生DenseNet结构进行轻量化适配：调整首层卷积核尺寸与步长，删减原始网络冗余的初始最大池化层，降低过渡层下采样倍率、适度减小网络增长率与模块深度，适配小尺寸图像特征提取，全程完整保留**密集块+跨层密集连接**核心机制，保证模型原理一致性。（基于标准DenseNet-BC轻量化结构，结合CIFAR-10数据集特性简化网络层级，核心融合逻辑完全一致）

## 数据集
我们使用的是数据集CIFAR-10，是一个更接近普适物体的彩色图像数据集。CIFAR-10是由Hinton的学生Alex Krizhevsky和Ilya Sutskever整理的一个用于识别普适物体的小型数据集。一共包含10个类别的RGB彩色图片：飞机（airplane）、汽车（automobile）、鸟类（bird）、猫（cat）、鹿（deer）、狗（dog）、蛙类（frog）、马（horse）、船（ship）和卡车（truck）。每个图片的尺寸为32×32，每个类别有6000个图像，数据集中一共有50000张训练图片和10000张测试图片。
数据集链接为：https://www.cs.toronto.edu/~kriz/cifar.html

它不同于我们常见的图片存储格式，而是用二进制优化了储存，当然我们也可以将其复刻出来为PNG等图片格式，但那会很大，我们的目标是神经网络，这里不做细致解析数据集，如果你想了解该数据集请观看链接：https://cloud.tencent.com/developer/article/2150614

---

## Introduction
DenseNet (Densely Connected Convolutional Networks) is a milestone deep convolutional neural network proposed by Gao Huang and other researchers in 2017, published in *Densely Connected Convolutional Networks* and awarded the Best Paper at CVPR 2017. On the basis of Residual Network, it innovatively designs **Dense Block** and **Transition Layer**, and realizes dense feature transmission through layer-wise channel concatenation.

With the core strategy of dense feature reuse and multi-path gradient backpropagation, DenseNet effectively solves the gradient vanishing problem of ultra-deep networks, greatly reduces parameters and computational cost, and improves feature utilization and generalization. It has outstanding performance in ImageNet classification and various downstream vision tasks. As one of the two classic backbone networks alongside ResNet, DenseNet is widely used in object detection, semantic segmentation, medical image analysis and lightweight model research.

## Architecture
The core architecture of DenseNet is a segmented deep neural network with **alternating stacking of dense blocks and transition layers**. The overall structure consists of an **initial convolution layer**, **multi-stage dense modules**, **transition downsampling layers**, **global average pooling** and a **fully connected classification head**, breaking the traditional unidirectional single-layer feature transfer mode of CNNs.

The standard input of the original paper is 224×224 RGB images with 3 channels, covering multiple network specifications including DenseNet121/169/201/264. The core design is as follows:
- **Core Feature Unit (Dense Block)**: The basic component of DenseNet. Each convolutional layer in the block takes feature maps from all previous layers as input and fuses features through channel concatenation. A fixed Growth Rate controls the output channel of each layer to avoid dimension explosion and realize efficient dense feature stacking with low parameters.
- **Dense Connection Mechanism**: Different from the element-wise addition fusion of ResNet, DenseNet splices features by channel concatenation. Each layer is directly connected to all subsequent layers, and gradients are backpropagated through multiple paths, which significantly improves the trainability of deep networks.
- **Transition Layer & Multi-Scale Design**: Transition layers are arranged between adjacent dense blocks, using 1×1 convolution for channel compression and average pooling for downsampling to balance model complexity and receptive field. The network extracts low-level texture, mid-level details and high-level semantic features step by step. Global average pooling is used before classification to reduce parameters and suppress overfitting.

<img width="735" height="55" alt="image" src="https://github.com/user-attachments/assets/562e4e4d-a897-46e7-9932-f61b73edb323" />


Relying on the efficient design of dense feature reuse, DenseNet achieves stronger feature expression with fewer parameters and stable gradient flow. The dense connection design has become an important research direction of modern convolutional networks, laying a solid foundation for feature fusion and lightweight visual models.

<img width="784" height="633" alt="image" src="https://github.com/user-attachments/assets/1787970f-f1bc-4784-8b2c-215b4c2cfe0f" />

<img width="1205" height="265" alt="image" src="https://github.com/user-attachments/assets/2b1f1401-fafb-4792-8c5d-24e988bd5df3" />
<img width="1199" height="590" alt="image" src="https://github.com/user-attachments/assets/61a7c4c7-c39f-409e-a84f-d6046b91b310" />




**Note:** This experiment adopts the 10-class CIFAR-10 dataset. Considering the 32×32 low-resolution input (much smaller than the 224×224 input in the original paper), we have made targeted lightweight adaptation for DenseNet: adjusting the kernel size and stride of the first convolutional layer, removing the redundant initial max-pooling layer in the original network, reducing the downsampling ratio of transition layers and properly optimizing network growth rate and module depth. The core dense mechanism including dense blocks and cross-layer dense connections is completely retained to ensure the consistency of model principles. (Based on the standard DenseNet-BC structure, the network hierarchy is simplified for CIFAR-10 while retaining the core feature fusion logic.)

## Dataset
We used the CIFAR-10 dataset, a color image dataset that more closely approximates common objects. CIFAR-10 is a small dataset for recognizing common objects, compiled by Alex Krizhevsky and Ilya Sutskever. It contains RGB color images for 10 categories: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck. Each image is 32 × 32 pixels, with 6000 images per category. The dataset contains 50,000 training images and 10,000 test images.

The dataset link is: https://www.cs.toronto.edu/~kriz/cifar.html

It differs from common image storage formats, using binary-optimized storage. While we could recreate it as PNG or other image formats, that would result in a very large file size. Our focus is on neural networks, so we won't delve into a detailed analysis of the dataset here. If you'd like to learn more about this dataset, please see the link: https://cloud.tencent.com/developer/article/2150614

---
## 原文章 | Original article
Huang G, Liu Z, Van Der Maaten L, et al. Densely connected convolutional networks[C]//Proceedings of the IEEE conference on computer vision and pattern recognition. 2017.
