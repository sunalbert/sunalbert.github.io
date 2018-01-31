---
layout:	    post
title:      "Hack into Deep Models"
subtitle:   "Vanilla Backpropagation Model for Deep Model Visualization"
date:       2017-04-13
author:     "Jinquan"
header-img: "img/post-dynet-insight-bg.jpg"
tags:
    - 深度学习
    - 可视化

---

> 上一篇文章详细介绍了Batch Normalization的原理、涉及的数学公式，基于这些内容，我们可以实现一个简单的BN操作。但是在实际应用中，只是实现公式描述的BN是无法满足系统需求的。本文将以作者向dynet贡献BN源代码的经历，详细描述如何基于Eigen实现一个实用的Batch Normalization操作。

### 文章结构

1. [滑动均值](#滑动均值)
2. [滑动方差](#滑动方差)
3. [实现代码](#实现代码)

#### 滑动均值/方差

上一篇文章中讲到在训练过程中，BN操作是基于当前batch计算mean和variance的。在测试过程中，我们希望每个BN操作是基于整体数据的mean和variance，扫描所有数据求得mean和variance显然是不现实的，但是测试阶段又不能缺少这两个变量，那么该怎么办呢？

实际上，金融领域已经采取了一些在线方法来计算流数据的mean和variance，这些在线方法得到的mean和variance被称作[moving_average](https://en.wikipedia.org/wiki/Moving_average)和moving_variance.

#### 滑动的实现方法

待补充

#### 总结

基于eigen的实现方式：