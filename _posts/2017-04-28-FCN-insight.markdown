---
layout:	    post
title:      "An Insight into FCN"
subtitle:   "FCN论文笔记与tensorflow实现"
date:       2017-04-28
author:     "Jinquan"
header-img: "img/post-dynet-insight-bg.jpg"
tags:
    - 深度学习
    - 图像分割
    - FCN
---

> FCN是2015年CVPR的Best Paper，提出了一种End-to-End的 Semantic Segmentation的方法，简称FCN(Fully Convolutional Network)。FCN可以直接使用Segmentation的Ground Truth作为监督信息，通过训练得到一个能够做pixel wise预测的网络。



### 文章结构

1. [论文概览](#论文概览)
2. [分割面临的问题](#分割面临的问题)
3. [FCN的解决方法](#FCN的解决方法)
4. [TensorFlow实现方式详解](#TensorFlow实现方式详解)



### 论文概览

FCN是2015年CVPR的Best Paper，提出了一种End-to-End的 Semantic Segmentation的方法，简称FCN(Fully Convolutional Network)。FCN可以直接使用Segmentation的Ground Truth作为监督信息，通过训练得到一个能够做pixel wise预测的网络。下图是一个直观的FCN网络结构与输出示意图：

![FCN-architecture](G:\sunalbert.github.io\img\in-post\post-FCN-tensorflow\FCN-architecture.JPG)

<small class="img-hint">FCN示意图</small>

简单地说，FCN在CNN分类网络的基础上，去除了最后的全连接层，并在对应位置添加上新的卷积层，使得原网络变成一个全卷积网络，然后将特定层的feature map通过反卷积做upsampling，恢复到最初的图片尺寸大小，最终于ground truth做pixle-wise的损失计算。

### 分割面临的问题

在FCN发表之前，传统的基于CNN的分割方法也是对每个像素分类，简单地说，就是以像素为中心取一个指定大小的patch，我们通过大量的patch来做训练，使得当中心像素处在目标区域是，网络输出positive，否则，网络输出negtive。首先，这种方法的计算开销是很大的，需要在输入图像上采用滑动方式找出所有patch，然后再将这些patch通过网络，网络输出的记过会被整合成最终的分类输出。

这种方法的设计初衷在于使用这些小的patch促使神经网络去去获取中心像素的context，但是在从生活经验来看，这种patch的方式恰恰限制了神经网络的识别能力。以笔者所关注的医疗图像分割场景来看，大多数情况下，标注人可以借助一张完整的CT图像标注出肿瘤的准确位置，却很难判断一个32X32的patch是否位于肿瘤中心。造成前后两种结果的主要原因就是：不同的数据提供方式，带来了不同的标注视野。简言之，基于patch分类的分割方法，限制了网络的视野，网络缺少了对更大范围内(甚至全局)context的感知。

### FCN的解决方法

FCN网络舍弃了基于patch的分割方法，转而寻找方法去直接利用图像的全局信息。对于一幅图像而言，目前人类标注的、能够当做全局信息的数据主要包括：图像所包含的物体、物体所在的位置信息。
为了能够有效的利用这种信息，作者提出直接在传统的目标检测网络的基础上做物体的分割。以VGG网络为例，具体方法如下：
1. 全连接层全部转化为卷积层
   就是讲VGG结构中最后几个全连接层部都转化为卷积层，使得整个网络变成一个全卷积网络。
2. Upsampling
   Upsampling，常见的叫法是反卷积，但是反卷积这个名字容易让人误以为是卷积反向过程，其实更正式的叫法应该是转置卷积(TransposrConvolution)，对于反卷积的理解，可以参考我的下一篇博文，博文中我会进一步介绍目前常见的反卷积方式和反卷积核。
   FCN网络整体上所做的改动主要就是这些，但在这些改动的基础上却取得了很好的分割效果。FCN网络的loss设计直接就是softmax

#### FCN网的缺陷



