---
layout:	    post
title:      "1x1卷积"
subtitle:   "1x1卷积的探索"
date:       2017-02-15
author:     "Jinquan"
header-img: "img/post-dynet-insight-bg.jpg"
tags:
    - 深度学习
---

> 1x1卷积，乍一看上去，似乎没有多少实际意义，但是在实际使用过程中，1x1卷积起到了很大的作用，在resnet、yolo等网络结构中，都出现了1x1卷积操作，本文通过一些资料和笔者平时的实验感受讲述1x1卷积的具体作用。

1x1卷积第一次被引入是在[Network in Network](https://arxiv.org/pdf/1312.4400.pdf)的工作中，在该工作中，作者使用MLPConv来代替了传统卷积中的线性组合操作，使得整个网络在增加较少参数的条件上做到了实际意义上的更深。该工作中的MLP被解释为在传统的conv上继续做了cross channel parametric pooling，极大增加了网络的非线性程度。该论文的具体细节。欢迎移步[NIN阅读笔记](#).

GoogLeNet再次提及了1x1卷积，在这篇论文中使用1x1卷积的主要目的是降低通道维度和减少模型参数，并使得网络可以更深。在随后的工作中，Resnet、YOLO等都使用了1x1卷积。

### 增加非线性特征

在不改变feature map大小的前提下大幅增加非线性特征（尽管1x1是严格线性的，但是一般在1x1卷积函数后，都会跟上ReLU激活函数，这样就可以不改变feature map的前提下，增加网络的非线性特征）

### 降低通道维度

目前，1x1卷积最为直观的作用就是**降低通道维度，进而减少参数的个数**。

1. 坐标敏感的转换函数，也可以是认为是跨通道的信息交互和整合的手段
2. 对通道数降维或者升维，减少参数

### 1x1卷积对感受野大小的改造



需要进一步阅读的论文：

1. goolenet
2. network in network
3. [全连接变卷积](http://hyichao.github.io/cv/2016/11/08/convolutional-vs-fc.html)
4. [Inception in CNN](http://blog.csdn.net/stdcoutzyx/article/details/51052847)
5. [dropout](http://blog.csdn.net/stdcoutzyx/article/details/49022443)
6. [geeps](https://users.ece.cmu.edu/~hengganc/archive/paper/%5Beurosys16%5Dgeeps.pdf)
7. [ps](http://blog.csdn.net/cyh_24/article/details/50545780)
8. [掌门一对一](http://www.newsmth.net/nForum/#!article/ITjob/108077)
9. **batch的大小为什么会影响训练效果**
10. 多层感知机和全连接网络的异同点



## 参考文献

1. [caffe社区](http://www.caffecn.cn/?/question/136)
2. [blog_1](http://iamaaditya.github.io/2016/03/one-by-one-convolution/)