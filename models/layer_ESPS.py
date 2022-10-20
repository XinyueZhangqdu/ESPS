import tensorflow.compat.v1 as tf



# nn.Conv2d(in, out, kernel, stride, padding, groups, bias)
# 512X1024X12, 12, 12, 3, 2
def group_conv(x, group, nout, kSize, stride):
    print(f"group conv : group is {group}, nout is {nout}, ksize is {kSize}, stride is {stride}")

    batch, heigt, width, channels = x.shape
    assert nout % group == 0, "分组巻积输出通道数应该能被分组数整除"

    if group >= 2:
        print("group conv : True")

        out_channel_per_group = int(nout / group)            ##### 每一组的输出输通数 #####
        channels_per_group = int(channels.value / group)     ##### 每一组的输入输通数 #####

        for g in range(1, group+1):
            if g==1:
                g_i = x[:, :, :, :g*channels_per_group]
                temp_o = tf.layers.conv2d(g_i, out_channel_per_group, kSize, strides=stride, padding='SAME')

            elif g==group:
                g_i = x[:, :, :, (g-1)*channels_per_group:]
                o_i = tf.layers.conv2d(g_i, out_channel_per_group, kSize, strides=stride, padding='SAME')
                temp_o = tf.concat([temp_o, o_i], -1)

            else:
                g_i = x[:, :, :, (g-1)*channels_per_group:g*channels_per_group]
                o_i = tf.layers.conv2d(g_i, out_channel_per_group, kSize, strides=stride, padding='SAME')
                temp_o = tf.concat([temp_o, o_i], -1)

    else:
        print("group conv : False")
        temp_o = tf.layers.conv2d(x, nout, kSize, strides=stride, padding='SAME')

    return temp_o


# nn.Linear
# full_connection_layer(out, exp_size)
def full_connection_layer(x, out_dim, trainable=True):
    in_dim = x.get_shape().as_list()[-1]
    W = tf.get_variable(
        name='weight',
        shape=[in_dim, out_dim],
        dtype=tf.float32,
        initializer=tf.truncated_normal_initializer(stddev=0.1),
        trainable=trainable)
    b = tf.get_variable(
        name='bias',
        shape=[out_dim],
        dtype=tf.float32,
        initializer=tf.constant_initializer(0.0),
        trainable=trainable)
    return tf.add(tf.matmul(x, W), b)


# nn.BatchNormed(nIn, eps=1e-03, momentum=BN_moment))
# batch_normalize(x, is_training=trainable, decay=0.10, epsilon=1e-03, trainable=True)
def batch_normalize(x, is_training, decay=0.10, epsilon=0.001, trainable=True):
    def bn_train():
        batch_mean, batch_var = tf.nn.moments(x, axes=[0, 1, 2])
        train_mean = tf.assign(pop_mean, pop_mean * decay + batch_mean * (1 - decay))
        train_var = tf.assign(pop_var, pop_var * decay + batch_var * (1 - decay))
        with tf.control_dependencies([train_mean, train_var]):
            return tf.nn.batch_normalization(
                x, batch_mean, batch_var, beta, scale, epsilon)

    def bn_inference():
        return tf.nn.batch_normalization(
            x, pop_mean, pop_var, beta, scale, epsilon)

    dim = x.get_shape().as_list()[-1]
    beta = tf.get_variable(
        name='beta_1',
        shape=[dim],
        dtype=tf.float32,
        initializer=tf.truncated_normal_initializer(stddev=0.0),
        trainable=trainable)
    scale = tf.get_variable(
        name='scale',
        shape=[dim],
        dtype=tf.float32,
        initializer=tf.truncated_normal_initializer(stddev=0.1),
        trainable=trainable)
    pop_mean = tf.get_variable(
        name='pop_mean',
        shape=[dim],
        dtype=tf.float32,
        initializer=tf.constant_initializer(0.0),
        trainable=False)
    pop_var = tf.get_variable(
        name='pop_var',
        shape=[dim],
        dtype=tf.float32,
        initializer=tf.constant_initializer(1.0),
        trainable=False)
    return tf.cond(is_training, bn_train, bn_inference)


# nn.UpsamplingBilinear2d(scale_factor=avgsize)
# deconv_layer(S2block_act_conv1x1_2, [batchsize,height,width,channels])
def upsample_bilinear(x, output_shape):

    output = tf.image.resize_bilinear(
        images=x,
        size=output_shape,
        align_corners=False,
        name=None)

    return output

# def deconv_layer(x, filter_shape, output_shape, stride, trainable=True):
#     filter_ = tf.get_variable(
#         name='filter_',
#         shape=filter_shape,
#         dtype=tf.float32,
#         initializer=tf.contrib.layers.xavier_initializer(),
#         trainable=True)
#
#     outputshape = [output_shape[0].value, output_shape[1].value, output_shape[2].value, output_shape[3].value]
#     deconv= tf.nn.conv2d_transpose(
#         value=x,
#         filter=filter_,
#         output_shape=outputshape,
#         strides=[1, stride, stride, 1])
#     return deconv


# nn.PReLU(nIn)
# prelu(SqueezeBlock_dense_1, trainable=True)
def prelu(x, trainable=True):
    alpha_prelu = tf.get_variable(
        name='alpha_prelu',
        shape=x.get_shape()[-1],
        dtype=tf.float32,
        initializer=tf.constant_initializer(0.0),
        trainable=trainable)
    pos = tf.nn.relu(x)
    neg = alpha_prelu * (x - tf.abs(x)) * 0.5
    return pos + neg



# nn.AvgPool2d(avgsize, avgsize)
# avg_pooling_layer(x, avgsize, avgsize)
def avg_pooling_layer(x, ksize, strides):
    if isinstance(ksize, int):
        kernel_size=[1, ksize, ksize, 1]
    else:
        kernel_size=[1, ksize[0], ksize[1], 1]

    if isinstance(strides, int):
        stride=[1, strides, strides, 1]
    else:
        stride=[1, strides[0], strides[1], 1]

    avg = tf.nn.avg_pool(
        value=x,
        ksize=kernel_size,
        strides=stride,
        padding='SAME')

    return avg


############################################################################################################################


def channel_shuffle(x, groups):
    batchsize, height, width, num_channels, = x.shape                           # 10x128x256x12
    channels_per_group = num_channels // groups
    x = tf.reshape(x, [batchsize, groups, height, width, channels_per_group])   # 10x128x256x12 ==> 10x4x128x256x3
    x = tf.transpose(x,perm=[0,2,1,3,4])                                        # 10x4x128x256x3 ==> 10x128x4x256x3
    x = tf.reshape(x,[batchsize, height, width, -1])                            # 10x128x4x256x3 ==> 10x128x256x12
    return x



def SqueezeBlock(x, exp_size, batch_size, divide=4.0):
        batch, height, width, channels = x.shape
        out = avg_pooling_layer(x, (height, width), stride=1)
        out = tf.reshape(out, [batch_size, -1])

        if divide > 1:
            SqueezeBlock_dense_1 = full_connection_layer(out, int(exp_size / divide))
            SqueezeBlock_dense_2 = prelu(SqueezeBlock_dense_1,trainable=True)
            SqueezeBlock_dense_3 = full_connection_layer(SqueezeBlock_dense_2, exp_size)
            SqueezeBlock_dense = prelu(SqueezeBlock_dense_3, trainable=True)
        else:
            SqueezeBlock_dense_1 = full_connection_layer(out, exp_size)
            SqueezeBlock_dense = prelu(SqueezeBlock_dense_1, trainable=True)

        SqueezeBlock_out = tf.reshape(SqueezeBlock_dense, [batch, 1, 1, channels])
        result = SqueezeBlock_out * x

        return result



def BR(x, trainable, nOut):
    with tf.variable_scope('BR_bn'):
        BR_bn = batch_normalize(x, is_training=trainable, decay=0.10, epsilon=1e-03, trainable=True)
        BR_act = prelu(BR_bn)
    return BR_act



# CBR(x_img,3, 12, 3, 2,is_training)
def CBR(x, nIn, nOut, kSize, stride=1, trainable=True):
        CBR_conv = tf.layers.conv2d(x, nOut, kSize, strides=stride, padding = 'same')
        CBR_bn = batch_normalize(CBR_conv, is_training=trainable, decay=0.10, epsilon=1e-03, trainable=True)
        CBR_act = prelu(CBR_bn, trainable=True)
        return CBR_act

def residual_ARM(x, nIn, nOut, kSize, stride=1, trainable=True):
        print(f'eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee')
        print(f'eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee')
        print(f'eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee')
        print(f'x.shape[1:3]{x.shape[1:3]}')
        batchsize,height,width,channels = x.shape
        #global_pool =  avg_pooling_layer(x, (height, width), strides=1)
        #print(f'global_pool{global_pool.shape}')
        CBR_conv = tf.layers.conv2d(x, nOut, kSize, strides=stride, padding = 'same')
        print(f'CBR_conv{CBR_conv.shape}')

        #CBR_bn = batch_normalize(CBR_conv, is_training=trainable, decay=0.10, epsilon=1e-03, trainable=True)
        #print(f'cbr_bn{CBR_bn.shape}')
        CBR_sig = tf.nn.sigmoid(CBR_conv)
        print(f'CBR_sig{CBR_sig.shape}')
        result  = CBR_sig * x
        return result
def residual_ARM_fusion(x, nIn, nOut, kSize, stride=1, trainable=True):
        print(f'eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee')
        print(f'eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee')
        print(f'eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee')
        print(f'x.shape[1:3]{x.shape[1:3]}')
        batchsize,height,width,channels = x.shape

        global_pool =  avg_pooling_layer(x, (height, width), strides=1)
        orin_shape = global_pool.shape[-1]
        CBR_conv = tf.layers.conv2d(global_pool, orin_shape//2, kSize, strides=stride, padding = 'same')
        CBR_relu = tf.nn.relu(CBR_conv)
        CBR_conv = tf.layers.conv2d(CBR_relu, orin_shape, kSize, strides=stride, padding = 'same')
        CBR_sig = tf.nn.sigmoid(CBR_conv)
        print(f'CBR_sig{CBR_sig.shape}')
        result  = CBR_sig * x
        return result


# SEseparableCBR(lever1_output, 12, self.dim1=16, 3, is_training, batch_size=10,stride =2, divide=1)
# SEseparableCBR(level3_0_output=128x256x64, self.dim1+self.dim2=64, self.dim2=48, 3, is_training, batch_size=10, stride =2, divide=1)
def SEseparableCBR(x, nIn, nOut, kSize, is_training, batch_size, stride=1, divide=2.0):
        group = nIn
        SEseparableCBR_conv = group_conv(x, group, nIn, kSize, stride)      # 256x512x12, 12, 12, 3, 2
        ##################################################################################################
        print(f"SEseparableCBR_conv output : {SEseparableCBR_conv.shape}")
        ######### 128x256x12 #########
        #########  64x128x64 #########
        ##################################################################################################

        SEseparableCBR_conv_2 = tf.layers.conv2d(SEseparableCBR_conv, nOut, kernel_size=1, strides=1)
        ##################################################################################################
        print(f"SEseparableCBR_conv_2 output : {SEseparableCBR_conv_2.shape}")
        ######### 128x256x16 #########
        #########  64x128x48 #########
        ##################################################################################################

        with tf.variable_scope('bn',reuse=False):
            SEseparableCBR_bn = batch_normalize(SEseparableCBR_conv_2, is_training, decay=0.10, epsilon=1e-03, trainable=True)
        SEseparableCBR_act = prelu(SEseparableCBR_bn)

        return SEseparableCBR_act



def level2_0_add_channels(x,trainable,nIn, nOut, batchsize, config= [[3,1],[5,1]],add=True):
    group_n = len(config)
    n = int(nOut / group_n)
    n1 = nOut - group_n * n
    level2_0_add_channels_c1 = tf.layers.conv2d(x,n, 1, strides=1, padding = 'same')
    level2_0_add_channels_channel_shuffle = channel_shuffle(level2_0_add_channels_c1, group_n)
    return level2_0_add_channels_channel_shuffle



# S2block(S2module_c1=128x256x12, trainable, n=24, n+n1=24, config[i]=[3,1])
def S2block(x, trainable, nIn, nOut, config):

        kSize = config[0]
        avgsize = config[1]
        if avgsize>1:
            ######################################################################################################
            print(f"S2block avgsize : {x.shape}")                                   ######### 128x256x64 #########
            ######################################################################################################
            x = avg_pooling_layer(x, avgsize, avgsize)
            ######################################################################################################
            print(f"S2block avg_pooling_layer : {x.shape}")                         ######### 128x256x64 #########
            ######################################################################################################

        with tf.variable_scope('s2blockgc', reuse=False):
            S2block_down_res_conv1 = group_conv(x, nIn, nIn, kSize, 1)
            S2block_down_res_conv2 = batch_normalize(S2block_down_res_conv1, is_training=trainable, decay=0.10, epsilon=1e-03, trainable=True)

        S2block_act_conv1x1_1 = prelu(S2block_down_res_conv2, trainable=True)
        S2block_act_conv1x1_2 = tf.layers.conv2d(S2block_act_conv1x1_1, nOut, 1, strides=1, padding='same')

        if avgsize>1:
            batchsize, height, width, channels = S2block_act_conv1x1_2.shape
            S2block_act_conv1x1_2 = upsample_bilinear(S2block_act_conv1x1_2, (avgsize*height,avgsize*width))
            ######################################################################################################
            print(f"S2block upsample_bilinear : {S2block_act_conv1x1_2.shape}")          ######### 128x256x64 #########
            ######################################################################################################
            # S2block_bn= batch_normalize(S2block_act_conv1x1_2_1, is_training=trainable, decay=0.10, epsilon=1e-03, trainable=True)

        S2block_bn= batch_normalize(S2block_act_conv1x1_2, is_training=trainable, decay=0.10, epsilon=1e-03, trainable=True)

        return S2block_bn



# S2module(level2_0=128x256x16, is_training, self.dim1=16, self.dim2=48, batchsize=10, config=config[i], add=False)
# config = [[3, 1], [5, 1]],
# S2module(level2_ModuleList=128x256x48, is_training, self.dim2=48, self.dim2=48, batchsize=10, config=config[i])
# config = [[3, 1], [3, 1]]
# S2module(output3_0=64x128x48, is_training, self.dim2=48, self.dim3=96, batchsize=10, config=config[2+i], add=False)
# config = [[3, 1], [5, 1]],
def S2module(x, trainable, nIn, nOut, batchsize, config= [[3,1],[5,1]], add=True):
        input = x
        group_n = len(config)               # 2     2       2
        n = int(nOut / group_n)             # 24    24      48
        n1 = nOut - group_n * n             # 0     0       0

        S2module_c1 = group_conv(x, group_n, n, 1, stride=1)
        S2module_channel_shuffle = channel_shuffle(S2module_c1, group_n)
        ##################################################################################################
        print(f"S2module_channel_shuffle output : {S2module_channel_shuffle.shape}")
        ######### 128x256x24 #########
        ######### 128x256x24 #########
        ######### 64x128x48  #########
        ##################################################################################################

        S2module_group ={}
        with tf.variable_scope('s2blocklist', reuse = False):
            # 0-2
            for i in range(group_n):
                with tf.variable_scope('s2blocklist{}'.format(i+1), reuse = False):
                    if i == 0:
                        S2module_group['{}'.format(i+1)] = S2block(S2module_c1, trainable, n, n+n1, config[i])
                        ###############################################################################################
                        print(f"S2module_group[{i+1}] output : {S2module_group[f'{i+1}'].shape}")
                        #### 128x256x24 ####
                        #### 128x256x24 ####
                        #### 64x128x48  ####
                        ###############################################################################################

                    else:
                        S2module_group['{}'.format(i+1)] = S2block(S2module_group['{}'.format(i)], trainable, n, n, config[i])
                        ###############################################################################################
                        print(f"S2module_group[{i+1}] output : {S2module_group[f'{i+1}'].shape}")
                        #### 128x256x24 ####
                        #### 128x256x24 ####
                        #### 64x128x48  ####
                        ###############################################################################################

        for i in range(group_n):
            result_d = S2module_group['{}'.format(i+1)]
            combine = result_d if i == 0 else tf.concat([combine,result_d], -1)

        if add:
            combine = input + combine

        with tf.variable_scope('BR_level',reuse=False):
            S2module_BR = BR(combine, trainable, nOut)

        return S2module_BR
