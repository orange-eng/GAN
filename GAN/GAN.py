from __future__ import print_function,division

from keras.datasets import mnist
from keras.layers import Input,Dense,Reshape,Flatten,Dropout
from keras.layers import BatchNormalization,Activation,ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D,Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam

import matplotlib.pyplot as plt
import numpy as np

import os
import sys
path = os.path.abspath(os.path.dirname(sys.argv[0]))

class GAN():
    def __init__(self):
        #mnist的shape 是 28*28*1
        self.img_row = 28
        self.img_col = 28
        self.channels = 1
        self.img_shape = (self.img_row,self.img_col,self.channels)
        self.latent_dim = 100
        #优化器
        
        optimizer = Adam(0.0002,0.5)
        #判别器
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',  #交叉熵
                                    optimizer=optimizer,
                                    metrics=['accuracy'])
        self.generator = self.build_generator()
        gan_input = Input(shape=(self.latent_dim,))
        img = self.generator(gan_input)

        #在训练generator的时候不可以训练discriminator
        self.discriminator.trainable = False
        #对生成的假图片进行预测
        validity = self.discriminator(img)          #根据generator生成的img，输出1或0
        self.combined = Model(gan_input,validity)
        self.combined.compile(loss='binary_crossentropy',optimizer=optimizer)

    #-------------------------------------------------------建立生成器输入为一组随机数字
    def build_generator(self):
        model = Sequential()
        model.add(Dense(256,input_dim=self.latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))

        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))

        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
               
        model.add(Dense(np.prod(self.img_shape),activation='tanh'))
        model.add(Reshape(self.img_shape))

        noise = Input(shape=(self.latent_dim,))     #输入是100个随机数字
        img = model(noise)                          #img为输出

        return Model(noise,img)
    
    #-----------------------------------------------判别器 对输入图片进行评价
    def build_discriminator(self):
        model = Sequential()
        model.add(Flatten(input_shape=self.img_shape))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        #判断真假 1为真，0为假
        model.add(Dense(1,activation='sigmoid'))

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img,validity)

    def train(self,epochs,batch_size=128,sample_interval=50):
        #获取数据
        (X_train,_),(_,_) = mnist.load_data()
        #标准化
        X_train = X_train / 127.5 -1
        X_train = np.expand_dims(X_train,axis=3)
        #创建标签
        valid = np.ones((batch_size,1))     #判断为真，则全部为1
        fake = np.zeros((batch_size,1))     #判断为假，则全部为0

        for epoch in range(epochs):
            #随机选取batch_size个图像，对discriminator进行训练
            idx = np.random.randint(0,X_train.shape[0],batch_size)  #随机生成28*28*1的随机数
            imgs = X_train[idx]

            noise = np.random.normal(loc=0,scale=1,size=(batch_size,self.latent_dim))    #正态分布生成一组数字

            gen_imgs = self.generator.predict(noise)

            d_loss_real = self.discriminator.train_on_batch(imgs,valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs,fake)
            d_loss = 0.5*np.add(d_loss_real,d_loss_fake)

            #训练generator
            noise = np.random.normal(loc=0,scale=1,size=(batch_size,self.latent_dim))
            g_loss = self.combined.train_on_batch(noise,valid)
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))
            if epoch % sample_interval == 0:
                self.sample_images(epoch)
    
    #展示图像
    def sample_images(self, epoch):

        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_imgs = self.generator.predict(noise)

        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig(path+"/images/%d.png" % epoch)
        plt.close()

if __name__ == '__main__':
    if not os.path.exists(path+"./images"):
        os.makedirs(path+"./images")
    gan = GAN()
    gan.train(epochs=401,batch_size=256,sample_interval=200)





