#train step1: 300 epoches for decoder SqueezeBlock
# zxy 2021.12.23
# modify by donghao at 2022.02.05

import os
import cv2
import json
import argparse
import numpy as  np
from PIL import Image
from tqdm import tqdm
import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt
from tensorflow.python import pywrap_tensorflow
#from keras.keras_flops import get_flops
from tensorflow.python.framework import graph_util
import load_imgMatting
from models.ESPS_zxy_b import ESPS_Decoder
import time
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def colormap_cityscapes(n,batchsize):
    cmap = np.zeros([n, 4]).astype(np.int32)
    #cmap[0, :] = np.array([128, 64, 128])
    #cmap[1, :] = np.array([244, 35, 232])
    cmap[0, :] = np.array([batchsize,0, 0, 0])
    cmap[1, :] = np.array([batchsize,255, 255, 255])
    #cmap[0, :] = np.array([150, 204, 244])
    #cmap[1, :] = np.array([248, 243, 228])
    #cmap[2, :] = np.array([0, 0, 0])
    '''
    cmap[2, :] = np.array([70, 70, 70])
    cmap[3, :] = np.array([102, 102, 156])
    cmap[4, :] = np.array([190, 153, 153])
    cmap[5, :] = np.array([153, 153, 153])

    cmap[6, :] = np.array([250, 170, 30])
    cmap[7, :] = np.array([220, 220, 0])
    cmap[8, :] = np.array([107, 142, 35])
    cmap[9, :] = np.array([152, 251, 152])
    cmap[10, :] = np.array([70, 130, 180])

    cmap[11, :] = np.array([220, 20, 60])
    cmap[12, :] = np.array([255, 0, 0])
    cmap[13, :] = np.array([0, 0, 142])
    cmap[14, :] = np.array([0, 0, 70])
    cmap[15, :] = np.array([0, 60, 100])

    cmap[16, :] = np.array([0, 80, 100])
    cmap[17, :] = np.array([0, 0, 230])
    cmap[18, :] = np.array([119, 11, 32])
    cmap[19, :] = np.array([0, 0, 0])
    '''
    return cmap

'''
class Colorize:

    def __init__(self, n=22):
        #self.cmap = colormap(256)
        self.cmap = colormap_cityscapes(256)
        self.cmap[n] = self.cmap[-1]
        self.cmap = self.cmap[:n]

    def __call__(self, gray_image):
        size = gray_image.shape
        #print(size)
        color_image = tf.zeros([3,size[0], size[1]],tf.int32)

        for label in range(0, len(self.cmap)):
            gray_image_data = gray_image[:,:,0]

            mask = tf.equal(gray_image_data,tf.cast(label,tf.bool))
            print(f'76 {self.cmap[label][0]}')
            cmap0 =  tf.fill([size[0],size[1]],self.cmap[label][0])
            tensor_cmap0 = tf.convert_to_tensor(cmap0,tf.int32)
            a = tf.where(mask,tensor_cmap0,color_image[0])

            cmap1 =  tf.fill([size[0],size[1]],self.cmap[label][1])
            tensor_cmap1 = tf.convert_to_tensor(cmap1,tf.int32)
            b = tf.where(mask,tensor_cmap1,color_image[1])
            cmap2 =  tf.fill([size[0],size[1]],self.cmap[label][2])
            tensor_cmap2 = tf.convert_to_tensor(cmap2,tf.int32)
            c = tf.where(mask,tensor_cmap2,color_image[2])

            color_image = tf.stack([a,b,c])
        return color_image
'''
class Colorize_batch:

    def __init__(self, n=22,batch_size =100):
        #self.cmap = colormap(256)
        self.cmap = colormap_cityscapes(256,batch_size)
        self.cmap[n] = self.cmap[-1]
        self.cmap = self.cmap[:n]

    def __call__(self, gray_image):
        size = gray_image.shape
        #print(size)
        color_image = tf.zeros([size[0],3,size[1], size[2]],tf.int32)
        #print(f'102{color_image.shape}')
        #print(f'self.cmap{self.cmap.size}')
        for label in range(0, len(self.cmap)):
            gray_image_data = gray_image[:,:,:,0]

            mask = tf.equal(gray_image_data,tf.cast(label,tf.bool))
            #print(f'value{self.cmap[label][0]}')
            cmap0 =  tf.fill([size[0],size[1],size[2]],self.cmap[label][1])
            tensor_cmap0 = tf.convert_to_tensor(cmap0,tf.int32)
            #print(f'111{color_image[:,0,:,:].shape}')

            a = tf.where(mask,tensor_cmap0,color_image[:,0,:,:])

            cmap1 =  tf.fill([size[0],size[1],size[2]],self.cmap[label][2])
            tensor_cmap1 = tf.convert_to_tensor(cmap1,tf.int32)
            b = tf.where(mask,tensor_cmap1,color_image[:,1,:,:])
            cmap2 =  tf.fill([size[0],size[1],size[2]],self.cmap[label][3])
            tensor_cmap2 = tf.convert_to_tensor(cmap2,tf.int32)
            c = tf.where(mask,tensor_cmap2,color_image[:,2,:,:])

            color_image = tf.stack([a,b,c],axis = 1)
        return color_image




def cout_flops(model):
    flops = tf.profiler.profile(model, options = tf.profiler.ProfileOptionBuilder.float_operation())
    print(f"flops {flops}")



def train_decoder():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--c','--config',type=str, default='./setting/ESPS_Decoder_s.json', help='JSON file for configuration')
    args = parser.parse_args()

    ################################ setting decoder framework ################################
    with open(args.c) as setting_encoder:
        config_encoder = json.load(setting_encoder)
    train_config = config_encoder['train_config']
    #start_learning_rate = train_config['learning_rate']
    #weight_decay = train_config['weight_decay']
    #weight_step = train_config['weight_step']
    model_name = train_config["Model"]
    #batch_size = train_config["batch"]
    batch_size = 1
    #epoch_all = train_config['epochs']
    data_config = config_encoder['data_config']
    data_dir = data_config["data_dir"]
    tf.reset_default_graph()

    edge_g =tf.placeholder(tf.float32, [batch_size, 224,224], name = 'edge_g')
    y_mask_g = tf.placeholder(tf.float32, [batch_size, 224, 224],name = 'y_mask_g')
    x_img_g = tf.placeholder(tf.float32, [batch_size, 224, 224,3], name = 'x_img_g')
    is_training = tf.placeholder(tf.bool, [],name="is_training")


    with tf.Session()as sess:
        assert model_name.startswith('Dnc_ESPS'), "Training function for the Decoder, please confirm model name."
        model = ESPS_Decoder(edge_g, y_mask_g, x_img_g, is_training, batch_size, train_config)
        #tensor_name = [tensor.name for tensor in tf.get_default_graph().as_graph_def().node]
        #print(tensor_name[0])
        #print(tensor_name[-1])

        ################################### calculate GFLOPs ###################################
        run_meta = tf.RunMetadata()
        opts = tf.profiler.ProfileOptionBuilder.float_operation()
        flops = tf.profiler.profile(graph = sess.graph, run_meta = run_meta, cmd = 'op', options = opts)
        params = tf.profiler.profile( sess.graph, options=tf.profiler.ProfileOptionBuilder.trainable_variables_parameter()) #计算参数量
        print('\n\n**********************************************************')
        print('Sum\t\t\t\t\t\t',flops.total_float_ops/1e9, 'Gfloat_ops')
        print('original params:',params.total_parameters)



if __name__=='__main__':
    train_decoder()
