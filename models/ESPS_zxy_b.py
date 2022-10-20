import numpy as np
import sys
import tensorflow as tf
import matplotlib.pyplot as plt

from layer_ESPS import *
import cv2
#tf.disable_v2_behavior()
config = [[[3, 1], [5, 1]],
          [[3, 1], [3, 1]],
          [[3, 1], [5, 1]],
          [[3, 1], [3, 1]],
          [[5, 1], [3, 2]],
          [[5, 2], [3, 4]],
          [[3, 1], [3, 1]],
          [[5, 1], [5, 1]],
          [[3, 2], [3, 4]],
          [[3, 1], [5, 2]]]


def _cumsum(x, axis=None):
    return np.cumsum(x, axis=axis)


def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    gts = tf.reduce_sum(gt_sorted)

    # intersection = gts - tf.cumsum(gt_sorted)
    intersection = gts - tf.py_func(_cumsum,[gt_sorted],tf.float32)
    # union = gts + tf.cumsum(1. - gt_sorted)
    union = gts + tf.py_func(_cumsum,[1. - gt_sorted],tf.float32)
    jaccard = 1. - intersection / union
    jaccard = tf.concat((jaccard[0:1], jaccard[1:] - jaccard[:-1]), 0)
    return jaccard


class ESPS_Encoder:
    def __init__(self, edge, y_mask, x_img, is_training, batch_size, train_config):
        self.batch_size = batch_size
        super().__init__()

        #################setting values
        self.classes = train_config["num_classes"]                  # 2
        self.p = 2
        self.q = 3
        self.chnn = train_config["chnn"]                            # 1

        print("ESPS Enc bracnch num :  " + str(len(config[0])))
        print("ESPS Enc chnn num:  " + str(self.chnn))

        self.dim1 = 16
        self.dim2 = 48 + 4 * (self.chnn - 1)
        self.dim3 = 96 + 4 * (self.chnn - 1)
        ################################generate the encoder module
        self.encoder_result = self.generator(x_img, is_training, self.batch_size)
        self.edge = edge
        self.mask = y_mask

        self.all_loss, self.loss, self.edgeloss = self.inferloss_iou(self.edge,self.mask, self.encoder_result)


    def generator(self, x_img, is_training,batchsize):

        with tf.variable_scope('encoder_generator',reuse = tf.AUTO_REUSE):
            with tf.variable_scope('lever1'):
                lever1_output = CBR(x_img,3, 12, 3, 2,is_training)
            print(f"lever1_output {lever1_output.shape}")

            with tf.variable_scope('level2_0'):
                level2_0 = SEseparableCBR(lever1_output, 12,self.dim1, 3, is_training,batch_size=batchsize, stride=2, divide=1)
            print(f"level2_0 {level2_0.shape}")

            with tf.variable_scope('level2'):
                for i in range(0,self.p):
                    with tf.variable_scope('level2_{}'.format(i), reuse = False):
                        if i ==0:
                            level2_ModuleList1 = S2module(level2_0,is_training,self.dim1, self.dim2,batchsize, config=config[i], add=False)
                            print(f"level2_ModuleList1{level2_ModuleList1.shape}")
                        else:
                            level2_ModuleList = S2module(level2_ModuleList1,is_training,self.dim2, self.dim2,batchsize, config=config[i])

            with tf.variable_scope('level3_0'):
                level3_0_data = tf.concat([level2_0,level2_ModuleList],-1)
                level3_0_output = BR(level3_0_data,is_training,self.dim1+self.dim2)

            with tf.variable_scope('level3_0',reuse = tf.AUTO_REUSE):
                output3_0 = SEseparableCBR(level3_0_output,self.dim1+self.dim2,self.dim2, 3,is_training,batch_size=batchsize,stride =2, divide=1)

            with tf.variable_scope('level3'):
                for i in range(0,self.q):
                    with tf.variable_scope('level3_{}'.format(i), reuse = False):
                        if i ==0:
                            level3_ModuleList = S2module(output3_0,is_training,self.dim2, self.dim3,batchsize, config=config[2+i], add=False)
                        else:
                            level3_ModuleList = S2module(level3_ModuleList,is_training,self.dim3, self.dim3,batchsize, config=config[2+i])
                level3_data_1 = tf.concat([output3_0,level3_ModuleList],-1)
                level3_0_output = BR(level3_data_1,is_training,self.dim1+self.dim2)

            with tf.variable_scope("classifier_encoder"):
                classifier_encoder_output = group_conv(level3_0_output,1,self.classes, 1,1)

        self.g_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'encoder_generator')
        return classifier_encoder_output


    def inferloss_iou(self, edge, mask, encoder_result):   #
        #criterion = lovasz_hinge(ignore = 255)
        encoder_result1 = encoder_result[:,:,:,0]
        encoder_result2 = encoder_result[:,:,:,1]

        loss = self.lovasz_hinge(encoder_result1,mask)
        #loss2 = self.lovasz_hinge(encoder_result2,mask)
        #loss = loss1+loss2

        #lossE1 = 0.5*self.lovasz_hinge(encoder_result1,edge)
        lossE = 0.2*self.lovasz_hinge(encoder_result2,edge)
        #lossE = lossE1+lossE2

        all_loss = loss + lossE
        return all_loss, loss, lossE

    def lovasz_hinge(self, logits, labels):

        qutanty = 1
        for i in range(0,len(logits.shape)):
            d = logits.shape[i]
            qutanty *= d
        logits = tf.reshape(logits, [qutanty])

        qutanty1 = 1
        for i in range(0,len(labels.shape)):
            d = labels.shape[i]
            qutanty1 *= d
        labels = tf.reshape(labels,[qutanty1])

        #ignore = 0
        signs = 2. * tf.to_float(labels) - 1.
        errors = (1. - logits * signs)

        perm = tf.argsort(errors, axis=0, direction = 'DESCENDING')
        errors_sorted = tf.gather(errors,perm)
        gt_sorted = tf.gather(labels,perm)
        sub, value = self.lovasz_grad(gt_sorted)
        relu_error = tf.nn.relu(errors_sorted)

        loss1 = tf.multiply(relu_error[1:qutanty1], sub)
        loss_a1 = tf.reduce_sum(loss1)
        loss2 = tf.multiply(relu_error[0:1], value)
        loss_a2 = tf.reduce_sum(loss2)
        loss_a = loss_a1 + loss_a2

        return loss_a


    def lovasz_grad(self, gt_sorted):
        #with tf.Session() as sess:

        q = 1
        for i  in range(0,len(gt_sorted.shape)):
            d = gt_sorted.shape[i]
            q *= d
        p = q
        gts =  tf.reduce_sum(gt_sorted)
        intersection = tf.to_float(gts) - tf.cumsum(tf.to_float(gt_sorted),0)
        union = tf.to_float(gts) + tf.cumsum(tf.to_float((1- gt_sorted)),0)
        jaccard = 1. - intersection / union
        v = tf.to_float(jaccard[1:q])
        c = tf.to_float(jaccard[0:-1])
        sub =tf.to_float( tf.subtract(v,c))
        value =tf.to_float(jaccard[0:1])

        return sub, value


class ESPS_Decoder:
    def __init__(self, edge, y_mask,x_img, is_training, batch_size, train_config):
        super().__init__()

        self.batch_size = batch_size
        ################# setting values
        self.p = train_config['p']
        self.q = train_config['q']
        self.chnn = train_config["chnn"]                            # 1
        self.classes = train_config["num_classes"]
        self.is_training = is_training
        print("ESPS Enc bracnch num :  " + str(len(config[0])))
        print("ESPS Enc chnn num:  " + str(self.chnn))

        self.dim1 = int((16)/2)
        self.dim2 =int(( 48 + 4 * (self.chnn - 1) )/2)                       # 48
        self.dim3 =int( (96 + 4 * (self.chnn - 1)  )/2 )                # 96

        self.img = x_img
        self.edge = edge
        self.mask = y_mask

        ################################ generate the encoder module
        self.decoder_result = self.generator(self.img, is_training, self.batch_size)
        self.all_loss, self.loss1, self.lossE = self.inferloss_iou(self.edge, self.mask, self.decoder_result)


    def generator(self, x_img, is_training, batchsize):
        print('222222222222222222222222')
        print('222222222222222222222222')
        with tf.variable_scope('encoder_generator', reuse = tf.AUTO_REUSE):
            with tf.variable_scope('lever1'):
                lever1_output = CBR(x_img, 3, 12, 3, 2, is_training)
                #lever1_output  =  SEseparableCBR(x_img, 3, 12, 3, is_training, batch_size=batchsize, stride=2, divide=1)
                ####################################################################################
                print(f"CBR output : {lever1_output.shape}\n")        ######### 256x512x12 #########
                ####################################################################################

            with tf.variable_scope('level2_0'):
                level2_0 = SEseparableCBR(lever1_output, 12, self.dim1, 3, is_training, batch_size=batchsize, stride=2, divide=1)
                ####################################################################################
                print(f"level2_0 output : {level2_0.shape}\n")        ######### 128x256x16 #########
                ####################################################################################

            with tf.variable_scope('level2'):
                ### self.p = 2 ###
                for i in range(0, self.p):
                    print(f"p = {i}")
                    with tf.variable_scope('level2_{}'.format(i), reuse = False):
                        if i ==0:
                            level2_ModuleList = S2module(level2_0, is_training, self.dim1, self.dim2, batchsize, config=config[i], add=False)
                            ######################################################################################################
                            print(f"level2_ModuleList output : {level2_ModuleList.shape}\n")        ######### 128x256x48 #########
                            ######################################################################################################
                        else:
                            level2_ModuleList = S2module(level2_ModuleList, is_training, self.dim2, self.dim2, batchsize, config=config[i])
                            ######################################################################################################
                            print(f"level2_ModuleList output : {level2_ModuleList.shape}\n")        ######### 128x256x48 #########
                            ######################################################################################################

            with tf.variable_scope('level3_0'):
                level3_0_data = tf.concat([level2_0, level2_ModuleList], -1)
                ######################################################################################################
                print(f"level3_0_data output : {level3_0_data.shape}")                  ######### 128x256x64 #########
                ######################################################################################################
                level3_0_output = BR(level3_0_data, is_training, self.dim1+self.dim2)
                ######################################################################################################
                print(f"level3_0_output output : {level3_0_output.shape}\n")            ######### 128x256x64 #########
                ######################################################################################################
            #with tf.variable_scope('level3_0_ARM1',reuse = tf.AUTO_REUSE):
            #    ARM_1 = residual_ARM(level3_0_output, 64, 64, 3, 1, is_training)
            #    print(f'ARM_1{ARM_1.shape}')
            with tf.variable_scope('level3_0_SE',reuse = tf.AUTO_REUSE):
                output3_0 = SEseparableCBR(level3_0_output, self.dim1+self.dim2, self.dim2, 3, is_training, batch_size=batchsize, stride =2, divide=1)
                ######################################################################################################
                print(f"output3_0 output : {output3_0.shape}\n")                        ######### 64x128x48 ##########
                ######################################################################################################

            with tf.variable_scope('level3_0_ARM_2',reuse = tf.AUTO_REUSE):
                ARM_2 = residual_ARM(output3_0, 24, 24, 3, 1, is_training)
                print(f'ARM_2{ARM_2.shape}')
                ARM_2_addChn = tf.layers.conv2d(ARM_2, 48, 3, strides=1, padding = 'same')

                #ARM_2_up = upsample_bilinear(ARM_2_addChn, (64, 128))
                #print(f'ARM_2{ARM_2_up.shape}')
            #with tf.variable_scope('level3_0_ARM_concat',reuse = tf.AUTO_REUSE):
                #ARM_concat = tf.concat([ARM_1, ARM_2_up], -1)
                #ARM_concat_cbr  = CBR(ARM_concat, 128, 128, 3, 1, is_training)
                #print(f'ARM_concat_cbr{ARM_concat_cbr.shape}')
                #ARM_concat_res =residual_ARM_fusion(ARM_concat_cbr, 128, 128, 3, 1, is_training)
                #print(f'ARM_concat_res{ARM_concat_res.shape}')
                #ARM_concat_res_changeChannel = tf.layers.conv2d(ARM_concat, 96, 3, strides=2, padding = 'same')
            with tf.variable_scope('level3'):
                ### self.q = 3 ###
                for i in range(0,self.q):
                    print(f"q = {i}")
                    with tf.variable_scope('level3_{}'.format(i), reuse = False):
                        if i ==0:
                            level3_ModuleList = S2module(output3_0, is_training, self.dim2, self.dim3, batchsize, config=config[2+i], add=False)
                            print(f"level3_ModuleList output : {level3_ModuleList.shape}\n")
                            level3_ModuleList = ARM_2_addChn + level3_ModuleList
                            ######################################################################################################
                            print(f"level3_ModuleList output : {level3_ModuleList.shape}\n")        ######### 64x128x96 ##########
                            ######################################################################################################
                        else:
                            level3_ModuleList = S2module(level3_ModuleList, is_training, self.dim3, self.dim3, batchsize, config=config[2+i])
                            ######################################################################################################
                            print(f"level3_ModuleList output : {level3_ModuleList.shape}\n")        ######### 64x128x96 ##########
                            ######################################################################################################

                level3_1_data = tf.concat([output3_0, level3_ModuleList], -1)
                ######################################################################################################
                print(f"level3_1_data output : {level3_1_data.shape}")                  ######### 64x128x144 #########
                ######################################################################################################
                level3_1_output = BR(level3_1_data, is_training, self.dim2+self.dim3)
                ######################################################################################################
                print(f"level3_1_output output : {level3_1_output.shape}\n")            ######### 64x128x144 #########
                ######################################################################################################

            with tf.variable_scope("classifier_encoder"):
                classifier_encoder_output = tf.layers.conv2d(level3_1_output, self.classes, 1, strides=1, padding = 'same')
                ################################################################################################################
                print(f"classifier_encoder_output output : {classifier_encoder_output.shape}")      ######### 64x128x2 #########
                ################################################################################################################


            #################### Decoder only ####################
            with tf.variable_scope('stage1_decoder'):
                batchsize, height, width, channels, = classifier_encoder_output.shape
                stage1_bn = upsample_bilinear(classifier_encoder_output, (height*2, width*2))
                #stage1_bn = deconv_layer(classifier_encoder_output, [1, 1,channels,channels], [batchsize,height*2,width*2,channels], 2)
                Dnc_stage1 = batch_normalize(stage1_bn, is_training=is_training, decay=0.99, epsilon=1e-03, trainable=True)
                #################################################################################################################
                print(f"Dnc_stage1 output : {Dnc_stage1.shape}")                                    ######### 128x256x2 #########
                #################################################################################################################

            with tf.variable_scope('confidence_decoder'):
                softmax_confidence = tf.nn.softmax(Dnc_stage1)
                stage1_confidence = tf.reduce_max(softmax_confidence, axis=-1)
                stage1_gate_unsqueeze = tf.expand_dims((1-stage1_confidence), axis=-1)
                stage1_gate  = tf.tile(stage1_gate_unsqueeze, multiples=(1,1,1,2))

            with tf.variable_scope('stage2_decoder'):
                Dnc_stage2_0 = CBR(level2_ModuleList, self.dim2, self.classes, 1, 1, trainable=is_training)
                value_bn = Dnc_stage2_0 * stage1_gate + Dnc_stage1
                #value_bn = stage1_gate + Dnc_stage1
                batchsize, height, width,channels = value_bn.shape
                Dnc_stage2_0_bn = upsample_bilinear(value_bn,(height*2,width*2))
                # Dnc_stage2_0_bn = deconv_layer(value_bn, [1, 1,channels,channels], [batchsize,height*2,width*2,channels], 2)
                Dnc_stage2 = batch_normalize(Dnc_stage2_0_bn, is_training=is_training, decay=0.99, epsilon=1e-03, trainable=True)
                #################################################################################################################
                print(f"Dnc_stage2 output : {Dnc_stage2.shape}")                                    ######### 256x512x2 #########
                #################################################################################################################

            with tf.variable_scope('classifier'):
                batchsize, height, width,channels = Dnc_stage2.shape
                classifier_1 = upsample_bilinear(Dnc_stage2, (height*2,width*2))
                # classifier_1 = deconv_layer(Dnc_stage2, [1, 1,channels,channels], [batchsize,height*2,width*2,channels], 2)
                classifier_2 = tf.layers.conv2d(classifier_1, self.classes, 3, strides=1, padding = 'same' ,name = 'ESPS_output')
                output_classifier = tf.cast(classifier_2,tf.float32,name = 'ESPS_zxy_model_result')
            self.g_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='encoder_generator')
            ######################################################################################################################
            print(f"classifier_2 output : {classifier_2.shape}\n")                                  ######### 512x1024x2 #########
            ######################################################################################################################

            return output_classifier


    def inferloss_iou(self, edge, mask, encoder_result):   #
        b, h, w, c = encoder_result.shape
        encoder_result1 = encoder_result[:,:,:,0]
        encoder_result2 = encoder_result[:,:,:,1]
        ignore = 0
        loss1 = self.lovasz_hinge(encoder_result1, mask, ignore)
        #loss2 = self.lovasz_hinge(encoder_result2, mask, ignore)
        lossE = 0.5*self.lovasz_hinge(encoder_result1,edge,ignore)

        #bce_loss1 = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels = mask, logits = encoder_result1))
        #bce_loss2 = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels = mask, logits = encoder_result2))
        #if self.is_training:
        #dice_loss1 = self.detail_Loss(encoder_result1, mask)
        #dice_loss2 = self.detail_Loss(encoder_result2, mask)

        #lossE = lossE1+lossE2

        #lossE = 0.5*loss

        all_loss = loss1 + lossE
        return all_loss, loss1, lossE
    def detail_Loss(self,logits,mask):
        #convert masks into edge maps

        laplacian_kernel = tf.convert_to_tensor([-1.,-1.,-1.,-1.,8.,-1.,-1.,-1.,-1.])
        laplacian_kernel = tf.reshape(laplacian_kernel,[3,3,1,1])

        gt_mask = tf.expand_dims(mask,-1)
        boundary_gt = tf.nn.conv2d(gt_mask,laplacian_kernel,padding = 'SAME')
        zero = tf.zeros_like(boundary_gt)
        ones = tf.ones_like(boundary_gt)
        boundary_gt = tf.where(boundary_gt<zero, zero, boundary_gt)
        compare_value = tf.constant(0.1,dtype = boundary_gt.dtype,shape = boundary_gt.shape)
        boundary_gt = tf.where(boundary_gt>compare_value,ones,zero)
        print(f"before calDice{logits.shape} mask{mask.shape} ")
        #calculate dice_loss

        dice_loss = self.dice_loss_func(tf.nn.sigmoid(logits),boundary_gt)
        return dice_loss
    def dice_loss_func(self, input, target):
        smooth = 1.
        n = input.shape[0]

        iflat = tf.reshape(input,[n,-1])
        tflat = tf.reshape(target,[n,-1])
        intersection = tf.reduce_sum((iflat*tflat),1)
        loss =  1 - ((2. * intersection + smooth)/ (tf.reduce_sum(iflat,1)+tf.reduce_sum(tflat,1) + smooth))
        return tf.reduce_mean(loss)
    def lovasz_hinge(self, logits, labels, per_image=None, ignore=255):
        """
        Binary Lovasz hinge loss
          logits: [B, H, W] Variable, logits at each pixel (between -\infty and +\infty)
          labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
          per_image: compute the loss per image instead of per batch
          ignore: void class id
        """
        # if per_image is not None:
        #     def treat_image(log_lab):
        #         log, lab = log_lab
        #         log, lab = tf.expand_dims(log, 0), tf.expand_dims(lab, 0)
        #         log, lab = flatten_binary_scores(log, lab, ignore)
        #         return lovasz_hinge_flat(log, lab)
        #     losses = tf.map_fn(treat_image, (logits, labels), dtype=tf.float32)
        #     loss = tf.reduce_mean(losses)
        # else:
        loss = self.lovasz_hinge_flat(*(self.flatten_binary_scores(logits, labels, ignore)))
        return loss


    def lovasz_hinge_flat(self, logits, labels):
        """
        Binary Lovasz hinge loss
          logits: [P] Variable, logits at each prediction (between -\infty and +\infty)
          labels: [P] Tensor, binary ground truth labels (0 or 1)
          ignore: label to ignore
        """


        def compute_loss():

            labelsf = tf.cast(labels, logits.dtype)
            signs = 2. * labelsf - 1.
            errors = 1. - logits * tf.stop_gradient(signs)
            errors_sorted, perm = tf.nn.top_k(errors, k=tf.shape(errors)[0], name="descending_sort")
            gt_sorted = tf.gather(labelsf, perm)
            grad = lovasz_grad(gt_sorted)
            loss = tf.tensordot(tf.nn.relu(errors_sorted), tf.stop_gradient(grad), 1, name="loss_non_void")
            return loss

        # deal with the void prediction case (only void pixels)
        loss = tf.cond(tf.equal(tf.shape(logits)[0], 0),
                       lambda: tf.reduce_sum(logits) * 0.,
                       compute_loss,
                       strict=True,
                       name="loss" )
        return loss


    def flatten_binary_scores(self, scores, labels, ignore=None):
        """
        Flattens predictions in the batch (binary case)
        Remove labels equal to 'ignore'
        """
        scores = tf.reshape(scores, (-1,))
        labels = tf.reshape(labels, (-1,))
        print('1111111111111111111111111111111111111111111111111111111')
        print(f'scores {scores.shape} labels {labels.shape}')
        if ignore is None:
            return scores, labels
        #return scores, labels

        valid = tf.not_equal(labels, ignore)
        vscores = tf.boolean_mask(scores, valid, name='valid_scores')
        vlabels = tf.boolean_mask(labels, valid, name='valid_labels')
        return vscores, vlabels
