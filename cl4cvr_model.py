#!/usr/bin/python
# -*- coding: utf-8 -*-
# with projection layer -> output proj vec

import numpy as np
import tensorflow as tf
import time
from flag_define import *

def get_time_stamp():
  ct = time.time()
  local_time = time.localtime(ct)
  data_head = time.strftime("%Y-%m-%d %H:%M:%S", local_time)
  data_secs = (ct - int(ct)) * 1000
  time_stamp = "%s.%03d" % (data_head, data_secs)
  return time_stamp

class Model:
    def get_initializer(self, mode, name):
        if mode != 'train':
            return tf.zeros_initializer()
        if name == 'he':
            return tf.keras.initializers.he_uniform()
        elif name == 'truncated_normal':
            return tf.truncated_normal_initializer(mean = 0.0, stddev = 0.01)
        elif name == 'glorot':
            return tf.glorot_uniform_initializer()
        raise ValueError('initializer name %s does not exist' % name)

    def get_optimizer(self, optimizer, loss, learning_rate):
        if optimizer == "Adagrad":
                    return tf.train.AdagradOptimizer(learning_rate).minimize(loss)
        elif optimizer == "Adam":
            return tf.train.AdamOptimizer(learning_rate).minimize(loss)
        else:
            raise ValueError('optimizer name %s does not exist' % optimizer)

    # input: (?, n_slot, k)
    # output: (?, 1)
    def get_dnn_output(self, data_embed_concat):
        layer_dim = []
        layer_dim_str = FLAGS.layer_dim_str.split(':')
        for item in layer_dim_str:
            layer_dim.append(item)

        n_layer = len(layer_dim)
#         keep_prob = 1.0

        cur_layer = data_embed_concat
        # loop to create DNN struct
        for i in range(0, n_layer):
            cur_layer = tf.layers.dense(cur_layer, units=layer_dim[i], activation=tf.nn.relu,
                        kernel_initializer=tf.glorot_uniform_initializer())
        # output layer, linear activation
        cur_layer = tf.layers.dense(cur_layer, units=1,
                    kernel_initializer=tf.glorot_uniform_initializer())

        y_hat = cur_layer
        return y_hat

    # input: (?, n_slot, k)
    # output: (?, 1)
    def get_dnn_last_layer_w_name(self, data_embed_concat, tower_name):
        layer_dim = []
        layer_dim_str = FLAGS.layer_dim_str.split(':')
        for item in layer_dim_str:
            layer_dim.append(item)

        n_layer = len(layer_dim)

        cur_layer = data_embed_concat
        # loop to create DNN struct
        for i in range(0, n_layer):
            cur_name = tower_name + str(i)
            cur_layer = tf.layers.dense(cur_layer, units=layer_dim[i], name=cur_name,
                        activation=tf.nn.relu,
                        kernel_initializer=tf.glorot_uniform_initializer())
        last_layer = cur_layer
        return last_layer

  # input_tensor_dict: labels & raw input data, in SparseTensor format
#     params = {
#       'sparse_columns': sparse_feature_columns,
#       'metrics_writer': metrics_writer
#     }
# mode: train, eval, predict

    def cl4cvr_loss(self, embedding, net1, net2, label, tau):
      # embedding: the output of embedding layer
      # net1/net2: contrastive network output
      # label: cvr_label
      # tau: adjustable parameter
      LARGE_NUM = 1e9
      batch_size = tf.shape(net1)[0]
      masks = tf.one_hot(tf.range(batch_size), batch_size)
      # sample similarity
      thre = 0.9
      net_norm = tf.math.l2_normalize(embedding, -1)
      net_sim = tf.matmul(net_norm, net_norm, transpose_b=True)
      sample_masks = tf.cast(tf.greater(net_sim, thre), tf.float32)
      # calculate the similarity
      net1 = tf.math.l2_normalize(net1, -1)
      net2 = tf.math.l2_normalize(net2, -1)
      net1_large = net1
      net2_large = net2
      logits_aa = tf.matmul(net1, net1_large, transpose_b=True) / tau
      logits_aa = logits_aa - sample_masks * LARGE_NUM
      logits_bb = tf.matmul(net2, net2_large, transpose_b=True) / tau
      logits_bb = logits_bb - sample_masks * LARGE_NUM
      logits_ab = tf.matmul(net1, net2_large, transpose_b=True) / tau
      logits_ab = logits_ab - (sample_masks - masks) * LARGE_NUM
      logits_ba = tf.matmul(net2, net1_large, transpose_b=True) / tau
      logits_ba = logits_ba - (sample_masks - masks) * LARGE_NUM
      if label == None:
        labels = tf.one_hot(tf.range(batch_size), batch_size * 2)
        positive_num = tf.reshape(tf.reduce_sum(labels, axis=1), [-1, 1])
      else:
        ones_label = tf.ones_like(label)
        label_opp = ones_label - label
        new_one_hot = tf.one_hot(tf.range(batch_size), batch_size*2)
        opp_one_hot = tf.multiply(label_opp, new_one_hot)
        reshape_label = tf.reshape(label, [1, -1])[0]
        copy_label = reshape_label
        label_sub = tf.map_fn(lambda x: x - copy_label, reshape_label)
        # ab_cast: if label is the same, then set 1 as a positive sample pair
        ab_cast = tf.cast(tf.equal(label_sub, 0.0), dtype=tf.float32)
        # aa_cast: Since 1 on the diagonal represents itself, it is not a positive sample pairs, so we need to set 0 to consider it as negative sample pairs
        aa_cast = tf.ones_like(ab_cast) - masks
        aa_cast = tf.multiply(ab_cast, aa_cast)
        labels = tf.concat([ab_cast, aa_cast], axis=1)
        labels = tf.multiply(label, labels)
        # add the matrix of positive and negative sample pairs when label=0
        labels = labels + opp_one_hot
        # positive_num：number of positive sample pairs
        positive_num = tf.reshape(tf.reduce_sum(labels, axis=1), [-1, 1])
      ones_mask = tf.concat([sample_masks-masks, sample_masks], axis=1)
      """ calculate loss """
      softmax_ab = tf.nn.softmax(tf.concat([logits_ab, logits_aa], 1))
      softmax_ab = softmax_ab + ones_mask
      loss_ab = tf.reduce_sum(labels * tf.log(softmax_ab), axis=1)
      loss_ab = tf.reduce_mean(-1.0 * tf.div_no_nan(loss_ab, positive_num))

      softmax_ba = tf.nn.softmax(tf.concat([logits_ba, logits_bb], 1))
      softmax_ba = softmax_ba + ones_mask
      loss_ba = tf.reduce_sum(labels * tf.log(softmax_ba), axis=1)
      loss_ba = tf.reduce_mean(-1.0 * tf.div_no_nan(loss_ba, positive_num))
      loss = loss_ab + loss_ba
      return loss

    def model_fn(self, input_tensor_dict, mode, params):
        tf.set_random_seed(123)
        print('Time: %s | Build Input' % (get_time_stamp()))
        with tf.variable_scope('input',
                               initializer = self.get_initializer(mode, 'truncated_normal'), reuse=tf.AUTO_REUSE):
            dnn_sparse_input = tf.feature_column.input_layer(input_tensor_dict['sparse_features'], params['sparse_columns'])

        if params['dense_columns'] != None:
          with tf.variable_scope('dense_input', initializer = \
            self.get_initializer(mode, 'truncated_normal'), reuse=tf.AUTO_REUSE):
            dnn_dense_input = tf.feature_column.input_layer(input_tensor_dict['dense_features'], params['dense_columns'])
            net = tf.concat([dnn_sparse_input, dnn_dense_input], 1)
        else:
          net = dnn_sparse_input
        # data agument
        net_dim = tf.shape(net)[1]
        net_mask = tf.keras.backend.random_binomial(shape=[1, net_dim], p=0.5, dtype=tf.float32)
        net_mask = tf.reshape(net_mask, [1, -1])
        net_mask_1 = net_mask * net
        net_mask_2 = (1.0 - net_mask) * net

        print('Time: %s | Build CTR Net' % (get_time_stamp()))
        with tf.variable_scope('ctr',
                initializer = self.get_initializer(mode, 'he'), reuse=tf.AUTO_REUSE):
            ctr_logits = self.get_dnn_output(net)
        print('Time: %s | Build CVR Net' % (get_time_stamp()))
        with tf.variable_scope('cvr',
                initializer = self.get_initializer(mode, 'he'), reuse=tf.AUTO_REUSE):
            
            cvr_last_layer = self.get_dnn_last_layer_w_name(net, 'cvr')
            cvr_logits = tf.layers.dense(cvr_last_layer, units=1,
                 name='cvr_logits', kernel_initializer=self.get_initializer(mode, 'he'))

        print('Time: %s | Build Cntrastive Net' % (get_time_stamp()))
        with tf.variable_scope('contrastive',
                 initializer = self.get_initializer(mode, 'he'), reuse=tf.AUTO_REUSE):
            # projection
            proj_dim_1 = 256
            proj_dim_2 = 128
            proj_dim_3 = 64
            proj_cvr_last_layer_1_0 = tf.layers.dense(net_mask_1, 
                activation=tf.nn.relu, units=proj_dim_1,
                name='proj_1', kernel_initializer=self.get_initializer(mode, 'he'))
            proj_cvr_last_layer_2_0 = tf.layers.dense(net_mask_2, 
                activation=tf.nn.relu, units=proj_dim_1,
                name='proj_1', kernel_initializer=self.get_initializer(mode, 'he'))
            proj_cvr_last_layer_1_1 = tf.layers.dense(proj_cvr_last_layer_1_0,
                activation=tf.nn.relu, units=proj_dim_2,
                name='proj_2', kernel_initializer=self.get_initializer(mode, 'he'))
            proj_cvr_last_layer_2_1 = tf.layers.dense(proj_cvr_last_layer_2_0,
                activation=tf.nn.relu, units=proj_dim_2,
                name='proj_2', kernel_initializer=self.get_initializer(mode, 'he'))
            proj_cvr_last_layer_1 = tf.layers.dense(proj_cvr_last_layer_1_1, 
                units=proj_dim_3,
                name='proj_3', kernel_initializer=self.get_initializer(mode, 'he'))
            proj_cvr_last_layer_2 = tf.layers.dense(proj_cvr_last_layer_2_1, 
                units=proj_dim_3,
                name='proj_3', kernel_initializer=self.get_initializer(mode, 'he'))            

        ctr_preds = tf.sigmoid(ctr_logits)
        cvr_preds = tf.sigmoid(cvr_logits)
        ctcvr_preds = tf.multiply(tf.stop_gradient(ctr_preds), cvr_preds)
        ctr_label = input_tensor_dict['ctr_label']
        cvr_label = input_tensor_dict['cvr_label']

        if mode == 'train':
            print('Time: %s | Build Loss' % (get_time_stamp()))
            ctr_loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(labels = ctr_label, logits = ctr_logits))
            ctcvr_loss = tf.reduce_mean(
                tf.losses.log_loss(labels = cvr_label, predictions = ctcvr_preds, reduction=tf.losses.Reduction.NONE))
            contrastive_loss = self.cl4cvr_loss(net, proj_cvr_last_layer_1, proj_cvr_last_layer_2, cvr_label, 12.0) 
            cl_weight = 0.01
            loss = ctr_loss + ctcvr_loss + cl_weight*contrastive_loss
            
            print('Time: %s | Build Optimizer' % (get_time_stamp()))
            learning_rate = FLAGS.learning_rate
            optimizer = self.get_optimizer(FLAGS.optimizer, loss, learning_rate)
            _, ctr_auc_op = tf.metrics.auc(labels=ctr_label, predictions=ctr_preds, name="ctr_auc")
            _, cvr_auc_op = tf.metrics.auc(labels=cvr_label, predictions=cvr_preds, name="cvr_auc")
            return {'run_ops' : optimizer,
                   'train_info' : [loss, ctr_auc_op, cvr_auc_op],
                   'eval_info' : [ctr_label, ctr_preds, cvr_label, cvr_preds]
                   }
        elif mode == 'test':
            ctr_loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(labels = ctr_label, logits = ctr_logits))
            ctcvr_loss = tf.reduce_mean(
                tf.losses.log_loss(labels = cvr_label, predictions = ctcvr_preds, reduction=tf.losses.Reduction.NONE))
            return {'test_info' : [ctr_label, ctr_preds, cvr_label, cvr_preds, cvr_label, ctcvr_preds, ctr_loss, ctcvr_loss]}

