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

FCN发表于2015年CVPR会议