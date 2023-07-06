#!/usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import datetime
from sklearn import metrics
from flag_define import *

def cal_auc(pred_score, label):
    fpr, tpr, thresholds = metrics.roc_curve(label, pred_score, pos_label=1)
    auc_val = metrics.auc(fpr, tpr)
    return auc_val, fpr, tpr

def cal_group_auc(pred_score, label, uid):
    uid_dict = {}
    for i in range(len(uid)):
        cur_uid = uid[i]
        cur_score = pred_score[i]
        cur_label = label[i]

        if cur_uid not in uid_dict:
            u_pred_score_list = []
            u_label_list = []
            uid_dict[cur_uid] = [u_pred_score_list, u_label_list]
        val = uid_dict[cur_uid]
        val[0].append(cur_score)
        val[1].append(cur_label)
        uid_dict[cur_uid] = val

    auc_val_list = []
    for key in uid_dict:
        val = uid_dict[key]
        u_pred_score_list = val[0]
        u_label_list = val[1]
        fpr, tpr, _ = metrics.roc_curve(u_label_list, u_pred_score_list, pos_label=1)
        auc_val = metrics.auc(fpr, tpr)
#        if np.isnan(auc_val):
#           auc_val = 1.0
        auc_val_list.append(auc_val)

    group_auc = np.nanmean(auc_val_list)
    return group_auc

# 1 - pred/true
def cal_bias(pred_score, label):
    ctr = np.sum(label) / len(label)
    pctr = np.mean(pred_score)
    bias = 1.0 - pctr/(ctr+1e-6)
    return bias, ctr, pctr

def cal_rmse(pred_score, label):
    mse = metrics.mean_squared_error(label, pred_score)
    rmse = np.sqrt(mse)
    return rmse

def cal_rectified_rmse(pred_score, label, sample_rate):
    for idx, item in enumerate(pred_score):
        pred_score[idx] = item/(item + (1-item)/sample_rate)
    mse = metrics.mean_squared_error(label, pred_score)
    rmse = np.sqrt(mse)
    return rmse

# only works for 2D list
def list_flatten(input_list):
    output_list = [yy for xx in input_list for yy in xx]
    return output_list


def count_lines(file_name):
    num_lines = sum(1 for line in open(file_name, 'rt'))
    return num_lines


time_style = '%Y-%m-%d %H:%M:%S'
def print_time():
    now = datetime.datetime.now()
    time_str = now.strftime(time_style)
    print(time_str)

#####################
# add delimiter as param
#####################

# load training data from csv files
def get_input_from_table(file_names, mode, batch_size, num_epochs=1):
    # if shffule = True -> shuffle over files; otherwise, train files one by one
    file_name_queue = tf.train.string_input_producer(file_names, num_epochs=num_epochs, shuffle=False)
    reader = tf.TextLineReader()
    _, value = reader.read(file_name_queue)
    # min_after_dequeue defines how big a buffer we will randomly sample from
    # capacity must be larger than min_after_dequeue and the amount larger determines the max we
    # will prefetch
    min_after_dequeue = 5000
    capacity = min_after_dequeue + 3*batch_size
    if mode == "train":
        batched_value = tf.train.shuffle_batch([value], \
                batch_size=batch_size, capacity=capacity, min_after_dequeue=min_after_dequeue)
    elif mode == 'test':
        batched_value = tf.train.batch([value], \
                batch_size=batch_size, capacity=capacity, allow_smaller_final_batch=True)        
    return batched_value


# if you have multiple labels or prefix info, you should revise the code below
def parse_instance(batched_value, batch_size, sparse_feature_list, dense_feature_list):
  sparse_fea_len = len(sparse_feature_list)
  dense_fea_len = len(dense_feature_list)
  
  if FLAGS.has_lineid == True:
    # num of pre cols before fts
    n_pre = 3
    # instance: lineid, label_1, label_2, [sparse_fea, ...], [dense_fea, ...]
    record_defaults = [['0']] + [[0.0]]*2 + [['']] * sparse_fea_len + [['0']] * dense_fea_len
  else:
    n_pre = 2
    # instance: label_1, label_2, [sparse_fea, ...], [dense_fea, ...]
    record_defaults = [[0.0]]*2 + [['']] * sparse_fea_len + [['0']] * dense_fea_len      

  records = tf.decode_csv(batched_value, record_defaults = record_defaults, field_delim = ',')
  if FLAGS.has_lineid == True:
    lineids = tf.reshape(records[0], [-1, 1])
    ctr_labels = tf.reshape(records[1], [-1, 1])
    cvr_labels = tf.reshape(records[2], [-1, 1])
  else:
    ctr_labels = tf.reshape(records[0], [-1, 1])
    cvr_labels = tf.reshape(records[1], [-1, 1])

  sparse_feature_tensor = {}
  dense_feature_tensor = {}

  if len(records) != n_pre + sparse_fea_len + dense_fea_len:
    raise ValueError('table cols %d does not equal %d+%d+%d=$d !', len(records), n_pre, sparse_fea_len, dense_fea_len)    

  for i in range(sparse_fea_len):
#     feature_name = sparse_feature_list[i][0]
#     feature_slot = sparse_feature_list[i][1]
    sparse_col_string = tf.string_split(records[i + n_pre], delimiter = '|')

    sparse_feature_tensor[str(FLAGS.max_slot_num+i)] = sparse_col_string

    for i in range(dense_fea_len):
#       dense_feature_name = dense_feature_list[i][0]
      dense_feature_dim = int(dense_feature_list[i][1])
      # sparse tensor
      dense_value_string = tf.string_split(records[i+n_pre+sparse_fea_len], delimiter = '|')
      # convert to dense tensor
      dense_value_string = tf.sparse.to_dense(dense_value_string, default_value='')
      dense_col_float_val = tf.string_to_number(dense_value_string, out_type=tf.float32)
      dense_feature_tensor[str(FLAGS.max_slot_num*2+i)] = dense_col_float_val
  if FLAGS.has_lineid == True:      
    fn_dict = {'lineid': lineids,
               'ctr_label' : ctr_labels,
               'cvr_label' : cvr_labels,
               'sparse_features' : sparse_feature_tensor,
               'dense_features' : dense_feature_tensor
            }
  else:
    fn_dict = {'ctr_label' : ctr_labels,
               'cvr_label' : cvr_labels,
               'sparse_features' : sparse_feature_tensor,
               'dense_features' : dense_feature_tensor
            }
  return fn_dict

