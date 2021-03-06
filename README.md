# GAN
## 前言
* 这里是orange研究GAN（生成对抗网络）进行整理总结，总结的同时也希望能够帮助更多的小伙伴。后期如果有学习到新的知识也会与大家一起分享。

------
## 列表（后期会根据学习内容增加）
* GAN
    * MNIST手写数字识别
* DCGAN
    * MNIST手写数字识别


### 目录
1. [实现的内容 Achievement](#实现的内容)
2. [所需环境 Environment](#所需环境)

### 实现的内容
#### GAN使用MNIST实现手写数字识别
生成式对抗网络（GAN, Generative Adversarial Networks ）是一种深度学习模型，是近年来复杂分布上无监督学习最具前景的方法之一。

在GAN模型中，一般存在两个模块：
分别是生成模型（Generative Model）和判别模型（Discriminative Model）；二者的互相博弈与学习将会产生相当好的输出。

原始 GAN 理论中，并不要求生成模型和判别模型都是神经网络，只需要是能拟合相应生成和判别的函数即可。但实用中一般均使用深度神经网络作为生成模型和判别模型 。

一个优秀的GAN应用需要有良好的训练方法，否则可能由于神经网络模型的自由性而导致输出不理想。

其实简单来讲，一般情况下，GAN就是创建两个神经网络，一个是生成模型，一个是判别模型。

生成模型的输入是一行正态分布随机数，输出可以被认为是一张图片（或者其它需要被判定真伪的东西）。
判别模型的输入是一张图片（或者其它需要被判定真伪的东西），输出是输入进来的图片是否是真实的（0或者1）。

生成模型不断训练的目的是生成 让判别模型无法判断真伪的输出。
判别模型不断训练的的目的是判断出输入图片的真伪。

#### 1、Generator
生成网络的目标是输入一行正态分布随机数，生成mnist手写体图片，因此它的输入是一个长度为N的一维的向量，输出一个28,28,1维的图片。
#### 2、Discriminator
判别模型的目的是根据输入的图片判断出真伪。因此它的输入一个28,28,1维的图片，输出是0到1之间的数，1代表判断这个图片是真的，0代表判断这个图片是假的。
#### 训练思路
GAN的训练分为如下几个步骤：
1、随机选取batch_size个真实的图片。
2、随机生成batch_size个N维向量，传入到Generator中生成batch_size个虚假图片。
3、真实图片的label为1，虚假图片的label为0，将真实图片和虚假图片当作训练集传入到Discriminator中进行训练。
4、将虚假图片的Discriminator预测结果与1的对比作为loss对Generator进行训练（与1对比的意思是，如果Discriminator将虚假图片判断为1，说明这个生成的图片很“真实”）。

https://blog.csdn.net/weixin_44791964/article/details/103729797
#### DCGAN使用MNIST实现手写数字识别
DCGAN的全称是Deep Convolutional Generative Adversarial Networks ,
意即深度卷积对抗生成网络。

它是由Alec Radford在论文Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks中提出的。实际上它是在GAN的基础上增加深度卷积网络结构。
#### 1、Generator
生成网络的目标是输入一行正态分布随机数，生成mnist手写体图片，因此它的输入是一个长度为N的一维的向量，输出一个28,28,1维的图片。与普通GAN不同的是，生成网络是卷积神经网络。
#### 2、Discriminator
判别模型的目的是根据输入的图片判断出真伪。因此它的输入一个28,28,1维的图片，输出是0到1之间的数，1代表判断这个图片是真的，0代表判断这个图片是假的。与普通GAN不同的是，它使用的是卷积神经网络。

#### 训练思路
DCGAN的训练和GAN一样，分为如下几个步骤：
1、随机选取batch_size个真实的图片。
2、随机生成batch_size个N维向量，传入到Generator中生成batch_size个虚假图片。
3、真实图片的label为1，虚假图片的label为0，将真实图片和虚假图片当作训练集传入到Discriminator中进行训练。
4、将虚假图片的Discriminator预测结果与1的对比作为loss对Generator进行训练（与1对比的意思是，如果Discriminator将虚假图片判断为1，说明这个生成的图片很“真实”）。
https://blog.csdn.net/weixin_44791964/article/details/103743038

## 所需环境
* Anaconda3（建议使用）
* python3.6.6
* VScode 1.50.1 (IDE)
* pytorch 1.3 (pip package)
* torchvision 0.4.0 (pip package)
