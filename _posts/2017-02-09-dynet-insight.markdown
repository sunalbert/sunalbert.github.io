---
layout:     post
title:      "dynet初探"
subtitle:   "动态神经网络库dynet初探"
date:       2017-02-09
author:     "Jinquan"
header-img: "img/post-bg-unix-linux.jpg"
tags:
    - 深度学习
    - 深度学习框架
    - dynet
---

> 动态深度网络愈发受到业内关注，16年年末和17年年初，先后有dynet、pytorch和TensorFlow Flod三个动态深度框架发布。笔者从16年10月份开始关注dynet，逐步成为该项目的contributor，对这个深度框架整体设计和细节实现都较为了解。本篇即在笔者的源码阅读心得基础上加以整理归纳，介绍dynet的整体架构和设计思路。




## 文章结构

1. [整体介绍](#整体介绍)
   1. [项目介绍](#项目介绍)
   2. [项目架构](#项目架构)
2. [模块介绍](#模块介绍)
   1. [Tensor](#Tensor)
   2. [Node](#Node)
   3. [Expression](#Expression)
   4. [ComputationGraph](#ComputationGraph)
   5. [Model](#Model)
   6. [Trainer](#Trainer)

---



## 整体介绍


> 动态深度网络dynet，代码简洁，动态计算流图构建速度也较快。

### 项目介绍

[dynet](https://github.com/clab/dynet)是一个由**CMU clab**发起并主导开发的动态深度网络框架，专门针对自然语言处理上的深度学习做了优化和改进。dynet目前支持CPU和GPU两种计算模式，计划加入多GPU支持、模型并行，矩阵计算基于eigen，CPU上计算加速基于cblas。

### 项目架构

dynet将一个深度网络结构解构为如下几个重要模块：Tensor、Node、Expression、Model、ComputationGraph, SimpleEigen Trainer。每个模块具体含义和实现方式将会在下文做详细介绍。

---



## 模块介绍

#### Tensor

 在dynet中，训练过程，所有的数据，都被表示为Tensor，这些数据包括：输入数据、输出数据、网络参数等。Tensor可以说是组成dynet神经网络的基础。Tensor实现在tensor.h和tensor.cc两个文件中。具体实现方法依赖于eigen第三方矩阵计算工具。Tensor有如下成员变量：

```c++
Dim d;  //shape of tensor
float* v; // pointer to memory
std::<Tensor> bs;
Device* device;
DeviceMempool mem_pool;
```

dynet中的tensor实际是依赖于[eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page)实现的，dynet中的tensor只是对eigen::tensor的一次封装，这次封装为eigen::tensor进一步明确了维度、所在device、内存地址等信息。

#### Node

在dynet网络结构中，Node是网络的实际构建单位。Node既可以是由tensor组成的，也可以是由函数组成的，也就是说，有些node是tensor node，而有些node则是function node，在function node上要做一些计算，function node的输出又是一个tensor。但是有一点需要注意的是，实际上在组建深度网络的时候，我们并不会直接使用到node，取而代之的是建立在node之上的Expression类。通过Expression构件深度网络，简洁方便，而且符合人的书写方式，但是底层的以node组成网络的形式，又方便人类理解计算流图的结构。一个具体的计算流图如下所示：

![cg](/img/in-post/post-dynet-insight/cg.jpg)
<small class="img-hint">计算流图</small>

在dynet中，所有的参数型node都会被其所属的Model持有。Node基类的主要成员函数和成员变量有：
```C++
Dim dim_forward(); //计算输入数据的维度是否能够计算，并计算输出数据的维度
void forward_impl(); //前向传播的具体实现
void backward_impl(); // 后向传播的具体实现
void forward(); //前向传播对外调用接口，用于判断当前Node是否支持多批量计算
void backward();
unsigned artiy(); //此node的参数个数
vector<VariableIndex> args; //参数向量
Dim dim; //输出结果维度
Device* device; //Node计算所在的device CPU or GPU
```

#### Expression
Expression既是表达式的抽象表示，也是一些常用函数的抽象表示。Expression的基础是node，expression的管理者是ComputationGraph。在dynet中，深度网络由Expression组成，所有的参数在高层语义中都是Expression，只不过依赖于下层的node来做具体实现。下面以一个简单的例子，来看看dynet中，如何使用Expression，组成一个xor的深度网络：

```C++
// 创建一个新的网络，参数的持有者是model，所有参数都会保存在model中
const unsigned HIDDEN_SIZE = 8;
p_W = m.add_parameters({HIDDEN_SIZE, 2}); //返回的类型是parameter
p_b = m.add_parameters({HIDDEN_SIZE});
p_V = m.add_parameters({1, HIDDEN_SIZE});
p_a = m.add_parameters({1});
//深度网络参数表达式
Expression W = parameter(cg, p_W);
Expression b = parameter(cg, p_b);
Expression V = parameter(cg, p_V);
Expression a = parameter(cg, p_a);
//组建深度网络
vector<dynet::real> x_values(2);  // set x_values to change the inputs to the network
Expression x = input(cg, {2}, &x_values);
dynet::real y_value;  // set y_value to change the target output
Expression y = input(cg, &y_value);
Expression h = tanh(W*x + b);
Expression y_pred = V*h + a;
Expression loss_expr = squared_distance(y_pred, y);
```

上述代码阐明了在dynet中创建网络需要的若干主要步骤：

1. 向模型参加所有参数
2. 为参数创建表达式实例
3. 使用表达式构建深度网络（更直观的说，构建最终的目标函数）

前面交代过Node才是构建深度网络结构的基本单位，但是在上述代码中并未出现过任何Node类的实例，那么dynet是怎么从高层的Expression过渡到底层的Node呢。下面以Parameter w加入到深度网络的过程为例，大致介绍下其中的流程：

1. 将paramter转化为Expression (in xor.cc)

   ```C++
   Expression w = parameter(ctg, p_W);
   ```

2. 返回Expressin 实例化对象(in expr.cc)

   ```C++
   return Expression(&g, g.add_parameter(p))
   ```

3. 向网络中添加Node(in dynet.cc:add_parameter)

   ```C++
   VariableIndex ComputationGraph::add_parameters(Parameter p) {
     VariableIndex new_node_index(nodes.size());
     ParameterNode* new_node = new ParameterNode(p); // 创建新的Node
     nodes.push_back(new_node); // 将新Node添加到nodes向量中
     parameter_nodes.push_back(new_node_index); // w是参数，所以向parameter_nodes中也添加一次
     set_dim_for_new_node(new_node_index);
     return new_node_index;
   }
   ```

   ​



#### Apple iOS

iOS is a **Unix-like OS based on Darwin(BSD)** and OS X, which share some frameworks including Core Foundation, Founadtion and the Darwin foundation with OS X, but, Unix-like shell access is not avaliable for users and restricted for apps, **making iOS not fully Unix-compatible either.**

The iOS kernal is **XNU**, the kernal of Darwin.

#### XNU Kernel
XNU, the acronym(首字母缩写) for ***X is Not Unix***, which is the **Computer OS Kernel** developed at Apple Inc since Dec 1996 for use in the Mac OS X and released as free open source software as part of Darwin.

---

## Linux


> Linux is a Unix-like and mostly POSIX-compliant computer OS.


![Unix_timeline](http://upload.wikimedia.org/wikipedia/commons/thumb/c/cd/Unix_timeline.en.svg/800px-Unix_timeline.en.svg.png)


#### Linux Kernel

严格来讲，术语 Linux 只表示 [Linux Kernel](http://en.wikipedia.org/wiki/Linux_kernel) 操作系统内核本身，比如说 Android is Based on Linux (Kernel). Linus 编写的也只是这一部分，一个免费的 Unix-like Kernel，并不属于 GNU Project 的一部分。

但通常把 Linux 作为 Linux Kernel 与大量配合使用的 GNU Project Software Kit (包括 Bash, Lib, Compiler, 以及后期的 GUI etc) 所组合成的 OS 的统称。（包括各类 Distribution 发行版）

这类操作系统也被称为 **GNU/Linux**


#### GNU Project

The GNU Project is a **free software, mass collaboration** project, which based on the following freedom rights:

* Users are free to run the software, share (copy, distribute), study and modify it.
* GNU software guarantees these freedom-rights legally (via its license).
* So it is not only FREE but, more important, FREEDOM.

In order to ensure that the *entire* software of a computer grants its users all freedom rights (use, share, study, modify), even the most fundamental and important part, **the operating system**, needed to be written.

This OS is decided to called **GNU (a recursive acronym meaning "GNU is not Unix")**. By 1992, the GNU Project had completed all of the major OS components except for their kernel, *GNU Hurd*.

With the release of the third-party **Linux Kernel**, started independently by *Linus Torvalds* in 1991 and released under the GPLv0.12 in 1992, for the first time it was possible to run an OS **composed completely of free software**.

Though the Linux kernel is not part of the GNU project, it was developed using GCC and other GNU programming tools and was released as free software under the GPL.

Anyway, there eventually comes to the **GNU/Linux**


* **GPL**: GNU General Public License
* **GCC**: GNU Compiler Collection

其他与 GPL 相关的自由/开源软件公共许可证：

* [Mozilla Public License](http://en.wikipedia.org/wiki/Mozilla_Public_License)
* [MIT License](http://en.wikipedia.org/wiki/MIT_License)
* [BSD Public License](http://en.wikipedia.org/wiki/BSD_licenses)
  * GPL 强制后续版本必须是自由软件，而 BSD 的后续可以选择继续开源或者封闭
* [Apache License](http://en.wikipedia.org/wiki/Apache_License)

![Public License](http://dl2.iteye.com/upload/attachment/0047/4142/d770c85a-49b7-3c7f-8ae2-cbb6451e00d8.png)

#### Android

Android is a mobile OS based on **Linux Kernel**, so it's definitely **Unix-like**.  

**Linux is under GPL so Android has to be open source**.
Android's source code is released by Google under open source licenses, although most Android devices ultimately ship with a combination of open source and proprietary software, including proprietary software developed and licensed by Google *(GMS are all proprietary)*  

#### Android Kernel

Android's kernel is based on one of the Linux kernel's long-term support (LTS) branches.   

**Android's variant of the Linux kernel** has further architectural changes that are implemented by Google outside the typical Linux kernel development cycle, and, certain features that Google contributed back to the Linux kernel. Google maintains a public code repo that contains their experimental work to re-base Android off the latest stable Linux versions.

Android Kernel 大概是 Linux Kernel 最得意的分支了，Android 也是 Linux 最流行的发行版。不过，也有一些 Google 工程师认为 Android is not Linux in the traditional Unix-like Linux distribution sense. 总之这类东西就算有各种协议也还是很难说清楚，在我理解里 Android Kernel 大概就是 fork Linux Kernel 之后改动和定制比较深的例子。


#### Android ROM

既然提到 Android 就不得不提提 Android ROM

ROM 的本义实际上是只读内存：  

**Read-only memory** (ROM) is a class of storage medium used in computers and other electronic devices. Data stored in ROM can only be modified slowly, with difficulty, or not at all, so it is **mainly used to distribute firmware (固件)** (software that is very closely tied to specific hardware, and unlikely to need frequent updates).

ROM 在发展的过程中不断进化，从只读演变成了可编程可擦除，并最终演化成了 Flash  

* PROM (Programmable read-only memory)
* EPROM (Erasable programmable read-only memory)
* EEPROM (Electrically erasable programmable read-only memory)
  * Flash memory (闪存)

Flash 的出现是历史性的，它不但可以作为 ROM 使用，又因其极高的读写速度和稳定性，先后发展成为U盘（USB flash drives）、移动设备主要内置存储，和虐机械硬盘几条街的固态硬盘（SSD），可以说这货基本统一了高端存储市场的技术规格。

所以我们平时习惯说的 ROM 其实还是来源于老单片机时代，那时的 ROM 真的是写了就很难（需要上电复位）、甚至无法修改，所以那时往 ROM 里烧下去的程序就被称作 firmware ，固件。久而久之，虽然技术发展了，固件仍然指代那些不常需要更新的软件，而 ROM 这个词也就这么沿用下来了。

所以在 wiki 里是没有 Android ROM 这个词条的，只有 [List of custom Android firmwares](http://en.wikipedia.org/wiki/List_of_custom_Android_firmwares)

> A custom firmware, also known as a custom ROM, ROM, or custom OS, is an aftermarket distribution of the Android operating system. They are based on the Android Open Source Project (AOSP), hence most are open-sourced releases, unlike proprietary modifications by device manufacturers.

各类 Android ROM 在 Android 词类下也都是属于 **Forks and distributions** 一类的。

所以我说，其实各类 Android ROM 也好，fork Android 之流的 YunOS、FireOS 也好，改了多少东西，碰到多深的 codebase ……**其实 ROM 和 Distribution OS 的界限是很模糊的**，为什么 Android 就不可以是移动时代的 Linux ，为什么 Devlik/ART 就不能是移动时代的 GCC 呢？

#### Chrome OS

Chrome OS is an operating system based on the **Linux kernel** and designed by Google to work with web applications and installed applications.

虽然目前只是个 Web Thin Client OS ，但是 RoadMap 非常酷……

* **Chrome Packaged Application** (Support working offline and installed)
* **Android App Runtime** (run Android applications natively...fxxking awesome)

平复一下激动的心情，还是回到正题来：

#### Chromium OS

Chrome OS is based on Chromium OS, which is the open-source development version of Chrome OS, which is a **Linux distribution** designed by Google.

For Detail, Chromium OS based on [Gentoo Linux](http://en.wikipedia.org/wiki/Gentoo_Linux), emm...