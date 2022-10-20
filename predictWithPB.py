import numpy as np
import tensorflow as tf
from tensorboard.plugins.beholder.im_util import read_image
from tensorflow.python.framework import graph_util
import cv2 as cv2
from tqdm import tqdm
import os
import time
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def colormap_cityscapes(n,batchsize):
    cmap = np.zeros([n, 4]).astype(np.int32)
    cmap[0, :] = np.array([batchsize,0, 0, 0])
    cmap[1, :] = np.array([batchsize,255, 255, 255])
    return cmap


class Colorize_batch:

    def __init__(self, n=22,batch_size =100):
        self.cmap = colormap_cityscapes(256,batch_size)
        self.cmap[n] = self.cmap[-1]
        self.cmap = self.cmap[:n]

    def __call__(self, gray_image):
        size = gray_image.shape
        color_image = tf.zeros([size[0],3,size[1], size[2]],tf.int32)
        for label in range(0, len(self.cmap)):
            gray_image_data = gray_image[:,:,:,0]
            mask = tf.equal(gray_image_data,tf.cast(label,tf.bool))
            cmap0 =  tf.fill([size[0],size[1],size[2]],self.cmap[label][1])
            tensor_cmap0 = tf.convert_to_tensor(cmap0,tf.int32)
            a = tf.where(mask,tensor_cmap0,color_image[:,0,:,:])
            cmap1 =  tf.fill([size[0],size[1],size[2]],self.cmap[label][2])
            tensor_cmap1 = tf.convert_to_tensor(cmap1,tf.int32)
            b = tf.where(mask,tensor_cmap1,color_image[:,1,:,:])
            cmap2 =  tf.fill([size[0],size[1],size[2]],self.cmap[label][3])
            tensor_cmap2 = tf.convert_to_tensor(cmap2,tf.int32)
            c = tf.where(mask,tensor_cmap2,color_image[:,2,:,:])
            color_image = tf.stack([a,b,c],axis = 1)
        return color_image


def freeze_graph_test(pb_path, image_path):
    with tf.Graph().as_default():
        output_graph_def = tf.GraphDef()
        with open(pb_path, "rb") as f:
            output_graph_def.ParseFromString(f.read())
            tf.import_graph_def(output_graph_def, name="")
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            input_image_tensor = sess.graph.get_tensor_by_name("x_img_g:0")
            is_training_tensor = sess.graph.get_tensor_by_name("is_training:0")
            # 定义输出的张量名称
            output_tensor_name = sess.graph.get_tensor_by_name("encoder_generator/classifier/ESPS_output/BiasAdd:0")
            r_w = 512
            r_h = 256
            b = 32
            color_transform = Colorize_batch(2,b)
            x_img_batch = []
            all_time =  0
            # 读取测试图片
            img_name_all = os.listdir(image_path)
            total_image_count = len(img_name_all)
            n_iter = int(total_image_count/b)
            for iter in tqdm(range(0,n_iter)):
                img_name = img_name_all[iter*b:(iter+1)*b]
                x_img_batch = []
                for img in img_name:
                    img_data1 = cv2.imread(os.path.join(image_path,img))
                    img_data =  cv2.resize(img_data1, (r_w,r_h))
                    x_img_batch.append(img_data/255)

                decoder_result = sess.run(output_tensor_name, feed_dict={input_image_tensor: x_img_batch,is_training_tensor:True})
                b,w,h,c = decoder_result.shape
                bool_r = tf.greater(decoder_result,tf.fill([b,w,h,c],0.0))
                color_change = color_transform(bool_r)
                color_change = tf.transpose(color_change,[0,2,3,1])
                color_change_result = color_change.eval()
                for idx in range(0,b):
                    cv2.imwrite('./result/'+img_name[idx], color_change_result[idx])




if __name__ == '__main__':
    out_pb_path = "./0318_last.pb"
    # 测试pb模型
    image_path = 'dataset/img_data/images/'
    freeze_graph_test(pb_path=out_pb_path, image_path=image_path)
