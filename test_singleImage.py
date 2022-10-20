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
import numpy as np
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
    start_learning_rate = train_config['learning_rate']
    weight_decay = train_config['weight_decay']
    weight_step = train_config['weight_step']
    model_name = train_config["Model"]
    batch_size = 1
    #epoch_all = train_config['epochs']
    data_config = config_encoder['data_config']
    data_dir = "/others/dataset/Adobe/image/"
    mask_dir = "/others/dataset/Adobe/mask/"
    edge_dir = "/others/dataset/Adobe/edge/"
    tf.reset_default_graph()

    edge_g =tf.placeholder(tf.float32, [batch_size, 224,224], name = 'edge_g')
    y_mask_g = tf.placeholder(tf.float32, [batch_size, 224, 224],name = 'y_mask_g')
    x_img_g = tf.placeholder(tf.float32, [batch_size, 224, 224,3], name = 'x_img_g')
    is_training = tf.placeholder(tf.bool, [])


    with tf.Session()as sess:
        assert model_name.startswith('Dnc_ESPS'), "Training function for the Decoder, please confirm model name."
        model = ESPS_Decoder(edge_g, y_mask_g, x_img_g, is_training, batch_size, train_config)

        '''
        ################################### calculate GFLOPs ###################################
        run_meta = tf.RunMetadata()
        opts = tf.profiler.ProfileOptionBuilder.float_operation()
        flops = tf.profiler.profile(graph = sess.graph, run_meta = run_meta, cmd = 'op', options = opts)
        print('\n\n**********************************************************')
        print('Sum\t\t\t\t\t\t',flops.total_float_ops/1e9, 'Gfloat_ops')

        '''
        # ################################ prepare date ################################
        x_img_dir = data_dir
        #y_label_dir = os.path.join(data_dir, 'resize_masks')
        #edge_dir = os.path.join(data_dir, 'resize_edges')
        # x_img_dir = os.path.join(data_dir, 'resize_images')
        # y_label_dir = os.path.join(data_dir, 'resize_masks')
        # edge_dir = os.path.join(data_dir, 'resize_edges')
        x_img_file = os.listdir(x_img_dir)
        total_image_count = len(x_img_file)
        n_iter = int(total_image_count/batch_size)
        if total_image_count <batch_size:
            n_iter = total_image_count
        print('\n\n**********************************************************')
        print(f"The total number of images is {total_image_count}")
        with tf.variable_scope('ESPSv'):
            global_steps = tf.Variable(tf.constant(0), name = 'global_step', trainable = False)

        decay_learning_rate = tf.train.exponential_decay(
                    learning_rate = start_learning_rate,
                    global_step = global_steps,
                    decay_steps = weight_step,
                    decay_rate = weight_decay,
                    staircase = True)
        train_op = tf.train.AdamOptimizer(decay_learning_rate).minimize(model.all_loss, global_step=global_steps, var_list=model.g_variables)

        # ################################ initialize the model ################################
        init = tf.global_variables_initializer()
        sess.run(init)
        print('\n\n**********************************************************')

        restore_dir = train_config['save_dir']
        #if tf.train.get_checkpoint_state(restore_dir):
        print("rrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr")
        print("rrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr")
        print("rrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr")
        saver = tf.train.Saver()
        saver.restore(sess, os.path.join(restore_dir, 'epoch64.ckpt'))
        ################################ Train the model ################################
        all_load_time = 0
        all_processing_time = 0
        all_converColor_time = 0
        color_transform = Colorize_batch(2,batch_size)
        #color_transform = Colorize(2)

        for i in tqdm(range(0, n_iter)):
            img_name = x_img_file[i*batch_size:(i+1)*batch_size]
            edge_batch = []
            x_img_batch = []
            y_mask_batch = []
            t0 = time.time()
            for img in img_name:
                print(f'{img}')
                img_data = cv2.imread(os.path.join(x_img_dir,img))
                #img_data = cv2.resize(img_data,(224,224))
                x_img_batch.append(img_data/255)
                #label_data = cv2.imread(os.path.join(y_label_dir,img),0)
                label_data = cv2.imread(os.path.join(mask_dir,img),0)
                #print(f'219 label_data {label_data.shape}')
                y_mask_batch.append(label_data/255)
                #edge_data = cv2.imread(os.path.join(edge_dir,img),0)
                edge_data = cv2.imread(os.path.join(edge_dir,img),0)
                #print(f'219 label_data {edge_data.shape}')
                edge_batch.append(edge_data/255)

            t1 = time.time()
            load_time = t1-t0
            all_load_time += load_time
            print(f'load data {load_time}')
            _, all_loss, loss1,lossE ,decoder_result, img, mask, edge = sess.run(
            [train_op, model.all_loss,model.loss1, model.lossE, model.decoder_result,
             model.img,model.mask,model.edge],
             feed_dict = {edge_g:edge_batch, y_mask_g:y_mask_batch, x_img_g:x_img_batch, is_training: True})
            t2 = time.time()
            processing_time = t2-t1
            all_processing_time += processing_time
            print(f'processing {processing_time}')
            '''
            import numpy as np
            result = decoder_result[0][:,:,0]
            if not os.path.exists('/others/dataset/PPM/resultTest2/'):
                print('not exists')
            print(img_name[0])
            print(result.shape)
            cv2.imwrite(os.path.join('/others/dataset/PPM/resultTest2/',img_name[0]),result*255)
            '''
            b,w,h,c = decoder_result.shape
            bool_r = tf.greater(decoder_result,tf.fill([b,w,h,c],0.2))
            color_change = color_transform(bool_r)
            #print(f'111111111111111111{color_change.shape}')
            color_change = tf.transpose(color_change,[0,2,3,1])
            color_change_result = color_change.eval()
            '''
            for idx in range(0,batch_size):

                cv2.imwrite('/others/dataset/PPM/resultTest3/'+img_name[idx], color_change_result[idx])
            '''
            for idx in range(0,b):
                IMG = color_change_result[idx]
                #import matplotlib.pyplot as plt
                #plt.imshow(IMG)
                #plt.show()
                th, im_th = cv2.threshold(IMG.astype(np.uint8), 220, 255, cv2.THRESH_BINARY_INV);

                #Copy the thresholded image.
                im_floodfill = im_th.copy()

                #Mask used to flood filling.
                #Notice the size needs to be 2 pixels than the image.
                h, w = im_th.shape[:2]
                mask = np.zeros((h + 2, w + 2), np.uint8)

                #Floodfill from point (0, 0)
                cv2.floodFill(im_floodfill, mask, (0, 0), 255);

                #Invert floodfilled image
                im_floodfill_inv = cv2.bitwise_not(im_floodfill)

                #Combine the two images to get the foreground.
                im_out = im_th | im_floodfill_inv
                height = im_out.shape[0]
                weight = im_out.shape[1]
                channels = im_out.shape[2]

                for row in range(height):
                    for col in range(weight):
                        b = im_out[row, col, 0]
                        g = im_out[row, col, 1]
                        r = im_out[row, col, 2]
                        #print(r,g,b)
                        if ([r,g,b]!=[255,255,255]):
                            im_out[row,col] = 0

                cv2.imwrite('/others/dataset/Adobe/resultTest4/'+img_name[idx], 255-im_out)
            t3 = time.time()
            convertColor_time = t3-t2
            all_converColor_time += convertColor_time
            print(f'convert color {convertColor_time}')


        print(f'all_load_time{all_load_time/total_image_count}')
        print(f'all_processing_time{all_processing_time/total_image_count}')
        print(f'all_converColor_time{all_converColor_time/total_image_count}')



if __name__=='__main__':
    train_decoder()
