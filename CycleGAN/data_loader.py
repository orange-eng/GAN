import cv2 
from glob import glob   
# glob是python自己带的一个文件操作相关模块，用它可以查找符合自己目的的文件，就类似于Windows下的文件搜索
# https://www.cnblogs.com/lovemo1314/archive/2011/04/26/2029556.html

import numpy as np 
import os
import sys
apath = os.path.abspath(os.path.dirname(sys.argv[0]))

#得到文件下面的所有文件目录。果然很方便
path = glob(apath+"/datasets/monet2photo/*")  
print(path)
class DataLoader():
    def __init__(self,dataset_name,img_res=(128,128)):
        self.img_res = img_res
        self.dataset_name = dataset_name

    def load_data(self,domain,batch_size=1,is_testing = False):
        data_type = "train%s"% domain if not is_testing else "test%s"% domain

        path = glob(apath+"/datasets/%s/%s/*"%(self.dataset_name,data_type))

        batch_images = np.random.choice(path,size=batch_size)
        imgs = []

        for img_path in batch_images:
            img = self.imread(img_path)
            
            img = cv2.resize(img,self.img_res)  #把图像变为128*128*3
            img = np.array(img)/127.5 - 1
            cv2.imshow("img",img)
            cv2.waitKey(0)
            imgs.append(img)

        return imgs

    def load_batch(self,batch_size=1,is_testing=False):
        data_type = "train" if not is_testing else "val"
        path_A = glob(apath +'./datasets/%s/%sA/*' % (self.dataset_name, data_type))
        path_B = glob(apath +'./datasets/%s/%sB/*' % (self.dataset_name, data_type))

        self.n_batches = int(min(len(path_A),len(path_B)) / batch_size )
        print("min:",int(min(len(path_A),len(path_B))))
        total_samples = self.n_batches * batch_size

        path_A = np.random.choice(path_A, total_samples, replace=False)
        path_B = np.random.choice(path_B, total_samples, replace=False)

        for i in range(self.n_batches - 1):
            batch_A = path_A[i*batch_size:(i+1)*batch_size]
            batch_B = path_B[i*batch_size:(i+1)*batch_size]
            imgs_A, imgs_B = [], []
            for img_A,img_B in zip(batch_A,batch_B):
                '''
                a = [1, 2, 3]
                b = [4, 5, 6]
                a_b_zip = zip(a, b)  # 打包为元组的列表,而且元素个数与最短的列表一致
                print("type of a_b_zip is %s" % type(a_b_zip))  # 输出zip函数的返回对象类型
                a_b_zip = list(a_b_zip)  # 因为zip函数返回一个zip类型对象，所以需要转换为list类型
                print(a_b_zip)
                '''
                img_A = self.imread(img_A)
                img_B = self.imread(img_B)

                img_A = cv2.resize(img_A,self.img_res)
                img_B = cv2.resize(img_B,self.img_res)
                
                imgs_A.append(img_A)
                imgs_B.append(img_B)

            imgs_A = np.array(imgs_A,dtype=np.float32)/127.5 - 1
            imgs_B = np.array(imgs_B,dtype=np.float32)/127.5 - 1

            yield imgs_A,imgs_B
# 带yield的函数是一个生成器，而不是一个函数了，
# 这个生成器有一个函数就是next函数，next就相当于“下一步”生成哪个数
# 这一次的next开始的地方是接着上一次的next停止的地方执行的




    #把BGR格式的图片转化为RGB格式的图片
    def imread(self,path):  
        img = cv2.imread(path)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        return img 

# if __name__ == "__main__":
#     Data = DataLoader(dataset_name="monet2photo")
#     for batch_i,(imgs_A,imgs_B) in enumerate(Data.load_batch(50)):
#         print(batch_i)


