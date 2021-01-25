from __future__ import print_function,division

from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.backend.tensorflow_backend import set_session
from keras.optimizers import Adam
from keras import backend as K
from keras.layers import *
from keras.models import *
import keras

from nets.resnet import get_resnet
from data_loader import DataLoader
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import datetime
import sys
import os

apath = os.path.abspath(os.path.dirname(sys.argv[0]))

class CycleGAN():
    def __init__(self):
        self.img_rows = 128
        self.img_cols = 128
        self.channels = 3
        self.img_shape = (self.img_rows,self.img_cols,self.channels)

        #载入数据
        self.dataset_name = "monet2photo"
        self.dataset_loader = DataLoader(dataset_name=self.dataset_name,img_res=(self.img_rows,self.img_cols))

        optimizer = Adam(lr=0.0002,beta_1=0.5)

        # patch
        patch = int(self.img_rows / 2**4)
        self.disc_patch = (patch,patch,1)

        #------------------------------#
        #   创建判别网络
        #------------------------------#
        self.d_A = self.build_discriminator()
        self.d_B = self.build_discriminator()
        self.d_B.summary()
        self.d_B.compile(
            optimizer=optimizer,
            loss = 'mse',
            metrics=['accuracy']
        )
        self.d_A.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['accuracy']
        )
        #------------------------------#
        #   创建生成网络
        #------------------------------#
        self.g_AB = self.build_generator()
        self.g_BA = self.build_generator()
        
        img_A = Input(shape=self.img_shape)
        img_B = Input(shape=self.img_shape)

        #生成假的图片
        fake_A = self.g_BA(img_B)
        fake_B = self.g_AB(img_A)
        #再次生成
        reconstr_A = self.g_AB(fake_B)
        reconstr_B = self.g_BA(fake_A)

        # 生成真实的A和B的ID
        img_A_id = self.g_BA(img_A)
        img_B_id = self.g_BA(img_B)

        # 评价模型是否为真的
        valid_A = self.d_A(fake_A)
        valid_B = self.d_B(fake_B)

        # 训练
        self.combined = Model(
            inputs=[img_A,img_B],
            outputs=[   valid_A,valid_B,
                        reconstr_A,reconstr_B,
                        img_A_id,img_B_id]
        )
        self.combined.compile(
            optimizer=optimizer,
            loss_weights=[  0.5,0.5,        #因为有6个损失函数，所以loss的权重也必须有六个
                            0.5,0.5,
                            0.5,0.5
            ],
            loss=[  'mse','mse',
                    'mse','mse',
                    'mse','mse']
        )

    def build_generator(self):
        model = get_resnet(self.img_rows,self.img_cols,self.channels)
        return model
    
    def build_discriminator(self):
        def conv2d(layer_input, filters, f_size=4, normalization=True):
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            if normalization:
                d = InstanceNormalization()(d)
            d = LeakyReLU(alpha=0.2)(d)
            return d

        img = Input(shape=self.img_shape)
        # 64,64,64
        d1 = conv2d(img, 64, normalization=False)
        # 32,32,128
        d2 = conv2d(d1, 128)
        # 16,16,256
        d3 = conv2d(d2, 256)
        # 8,8,512
        d4 = conv2d(d3, 512)
        # 对每个像素点判断是否有效
        # 64
        # 8,8,1
        validity = Conv2D(1, kernel_size=3, strides=1, padding='same')(d4)

        return Model(img, validity)        

    def scheduler(self,models,epoch):
        # 每经过100个epoch,学习率就变成原来的0.5
        if epoch % 20 == 0 and epoch !=0:
            for model in models:
                lr = K.get_value(model.optimizer.lr)
                K.set_value(model.optimizer.lr,lr*0.5)
            print("lr changed to {}"/format(lr*0.5))

    def train(self,batch_size=1,epochs=2,sample_interval=1):
        
        start_time = datetime.datetime.now()

        valid = np.ones((batch_size,)+self.disc_patch)      # 尺度为(1,8,8,1)
        fake = np.ones((batch_size,) + self.disc_patch)     # 尺度为(1,8,8,1)
        
        for epoch in range(epochs):
            print("epoch:",epoch)
            self.scheduler([self.combined,self.d_A,self.d_B],epoch)
            for batch_i,(imgs_A,imgs_B) in enumerate(self.dataset_loader.load_batch(batch_size)):
                print("batch_i:",batch_i)
                # ------------------ #
                #  训练生成模型
                # ------------------ #                
                g_loss = self.combined.train_on_batch([imgs_A, imgs_B],
                                                        [valid, valid,
                                                        imgs_A, imgs_B,
                                                        imgs_A, imgs_B])
                # ---------------------- #
                #  训练评价者
                # ---------------------- #
                # A->B的假图片
                fake_B = self.g_AB.predict(imgs_A)
                fake_A = self.g_BA.predict(imgs_B)
                # 判断真假图片，并以此进行训练
                dA_loss_real = self.d_A.train_on_batch(imgs_A, valid)
                dA_loss_fake = self.d_A.train_on_batch(fake_A, fake)
                dA_loss = 0.5 * np.add(dA_loss_real, dA_loss_fake)
                # 判断真假图片，并以此进行训练
                dB_loss_real = self.d_B.train_on_batch(imgs_B, valid)
                dB_loss_fake = self.d_B.train_on_batch(fake_B, fake)
                dB_loss = 0.5 * np.add(dB_loss_real, dB_loss_fake)

                d_loss = 0.5 * np.add(dA_loss, dB_loss)

                elapsed_time = datetime.datetime.now() - start_time

                print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %3d%%] [G loss: %05f, adv: %05f, recon: %05f, id: %05f] time: %s " \
                                                                        % ( epoch, epochs,
                                                                            batch_i, self.dataset_loader.n_batches,
                                                                            d_loss[0], 100*d_loss[1],
                                                                            g_loss[0],
                                                                            np.mean(g_loss[1:3]),
                                                                            np.mean(g_loss[3:5]),
                                                                            np.mean(g_loss[5:6]),
                                                                            elapsed_time))
                if batch_i % sample_interval == 0:
                    #self.sample_images(epoch, batch_i)
                    #if epoch % 5 == 0 and epoch != init_epoch:
                    print("sample_interval:",sample_interval)
                    self.d_A.save_weights(apath+"/weights/%s/d_A_epoch%d.h5" % (self.dataset_name, epoch))
                    self.d_B.save_weights(apath+"/weights/%s/d_B_epoch%d.h5" % (self.dataset_name, epoch))
                    self.g_AB.save_weights(apath+"/weights/%s/g_AB_epoch%d.h5" % (self.dataset_name, epoch))
                    self.g_BA.save_weights(apath+"/weights/%s/g_BA_epoch%d.h5" % (self.dataset_name, epoch))
    
    def sample_images(self, epoch, batch_i):
        os.makedirs('images/%s' % self.dataset_name, exist_ok=True)
        r, c = 2, 2

        imgs_A = self.dataset_loader.load_data(domain="A", batch_size=1, is_testing=True)
        imgs_B = self.dataset_loader.load_data(domain="B", batch_size=1, is_testing=True)

        fake_B = self.g_AB.predict(imgs_A)
        fake_A = self.g_BA.predict(imgs_B)

        gen_imgs = np.concatenate([imgs_A, fake_B, imgs_B, fake_A])

        gen_imgs = 0.5 * gen_imgs + 0.5

        titles = ['Original', 'Translated', 'Reconstructed']
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt])
                axs[i, j].set_title(titles[j])
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("images/%s/%d_%d.png" % (self.dataset_name, epoch, batch_i))
        plt.close()

if __name__ == "__main__":
    cyclegan = CycleGAN()
    cyclegan.train(batch_size=1,epochs=2)
    


