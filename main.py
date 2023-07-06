#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
import datetime
import func_define as func
from flag_define import *
from feature_define import SparseFeatures, DenseFeatures
import time
from cl4cvr_model import Model
import os
import shutil

def get_time_stamp():
  ct = time.time()
  local_time = time.localtime(ct)
  data_head = time.strftime("%Y-%m-%d %H:%M:%S", local_time)
  data_secs = (ct - int(ct)) * 1000
  time_stamp = "%s.%03d" % (data_head, data_secs)
  return time_stamp

def get_ops(model, input_tensor_dict, params, mode):
  if mode == 'train':
    print('Time: %s | Start Building Train Model!' % get_time_stamp())
    ops_dict = model.model_fn(input_tensor_dict, 'train', params)
    print('Time: %s | Building Train Model Done!' % get_time_stamp())
    return ops_dict
  elif mode == 'test':
    print('Start Building Test Model!')
    ops_dict = model.model_fn(input_tensor_dict, 'test', params)
    print('Building Test Model Done!')
    return ops_dict
  elif mode == 'save_emb':
    print('Start Building Save Embedding Model!')
    ops_dict = model.model_fn(input_tensor_dict, 'test', params)
    print('Building Save Embedding Model Done!')
  else:
    raise ValueError('invalid mode flag [%s]' % FLAGS.mode)

def read_data():
    train_file_name = []; test_file_name = []
    for item in FLAGS.train_file_name.split(":"):
        train_file_name.append(item)
    for item in FLAGS.test_file_name.split(":"):
        test_file_name.append(item)

    train_values = func.get_input_from_table(train_file_name, 'train', FLAGS.batch_size, FLAGS.train_epoch)
    test_values = func.get_input_from_table(test_file_name, 'test', FLAGS.test_batch_size, 1)
    return train_values, test_values

def get_auc(test_label_all, test_pred_score_all):
    test_label_re = func.list_flatten(test_label_all)
    test_pred_score_re = func.list_flatten(test_pred_score_all)
    print(test_label_re[:20])
    print(test_pred_score_re[:20])
    test_auc, _, _ = func.cal_auc(test_pred_score_re, test_label_re)
    # rounding
    test_auc = np.round(test_auc, 4)
    return test_auc

def train_test(ops_dict_train, ops_dict_test):
    # Launch the graph.
    config = tf.ConfigProto(log_device_placement=False)
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.3

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess, coord)

        func.print_time()
        print('Start train loop')

        t1 = time.time()

        epoch = -1
        try:
            while not coord.should_stop():
                epoch += 1
                sess.run(ops_dict_train['run_ops'])
                train_result = sess.run(ops_dict_train['train_info'])

                # record loss and accuracy every step_size generations
                if (epoch+1)%FLAGS.steps_print_train_info == 0:
                    train_loss_temp, train_ctr_auc_temp, train_cvr_auc_temp = train_result
 
                    auc_and_loss = [epoch+1, train_loss_temp, train_ctr_auc_temp, train_cvr_auc_temp]
                    auc_and_loss = [np.round(xx,4) for xx in auc_and_loss]
                    func.print_time()
                    print('Generation # {}. Train Loss: {:.4f}. Train CTR AUC: {:.4f}. Train CVR AUC: {:.4f}.'\
                          .format(*auc_and_loss))

        except tf.errors.OutOfRangeError:
            func.print_time()

            # whether to save the model
            if FLAGS.save_model_ind == True:
                saver = tf.train.Saver()
                save_path = saver.save(sess, FLAGS.model_saving_addr)
                print("Model saved in file: %s" % save_path)
            print('Done training -- epoch limit reached')
#
        train_time = time.time() - t1

        # load test data
        test_ctr_label_all = []
        test_ctr_pred_score_all = []
        test_ctr_loss_all = []

        test_cvr_label_all = []        
        test_cvr_pred_score_all = []

        test_ctcvr_label_all = []
        test_ctcvr_pred_score_all = []
        test_ctcvr_loss_all = []                

        t2 = time.time()
        try:
            while True:
                cur_ctr_label, cur_ctr_preds, cur_cvr_label, cur_cvr_preds, \
                cur_cvr_label, cur_ctcvr_preds, cur_ctr_loss, cur_ctcvr_loss = sess.run(ops_dict_test['test_info'])
                test_ctr_label_all.append(cur_ctr_label)
                test_ctr_pred_score_all.append(cur_ctr_preds)
                test_ctr_loss_all.append(cur_ctr_loss)
                
                test_cvr_label_all.append(cur_cvr_label)
                test_cvr_pred_score_all.append(cur_cvr_preds)
                
                test_ctcvr_label_all.append(cur_cvr_label)
                test_ctcvr_pred_score_all.append(cur_ctcvr_preds)
                test_ctcvr_loss_all.append(cur_ctcvr_loss)             

        except tf.errors.OutOfRangeError:
            func.print_time()
            print('Done testing -- epoch limit reached')
        finally:
            coord.request_stop()

        coord.join(threads)

        test_time = time.time() - t2
        
        # filter out valid CVR labels and predictions (when CTR label = 1)
        test_ctr_label_all = func.list_flatten(test_ctr_label_all)
        test_cvr_label_all = func.list_flatten(test_cvr_label_all)
        test_cvr_pred_score_all = func.list_flatten(test_cvr_pred_score_all)
        fil_cvr_label_all = [b for a,b in zip(test_ctr_label_all, test_cvr_label_all) if float(a)==1.0]
        fil_cvr_pred_score_all = [b for a,b in zip(test_ctr_label_all, test_cvr_pred_score_all) if float(a)==1.0]

        # calculate auc
        test_ctr_auc = get_auc(test_ctr_label_all, test_ctr_pred_score_all)
        test_cvr_auc = get_auc(fil_cvr_label_all, fil_cvr_pred_score_all)
        test_ctcvr_auc = get_auc(test_ctcvr_label_all, test_ctcvr_pred_score_all)
 
        # rounding
        test_ctr_auc = np.round(test_ctr_auc, 4)
        test_cvr_auc = np.round(test_cvr_auc, 4)
        test_ctcvr_auc = np.round(test_ctcvr_auc, 4)
        test_ctr_loss = np.round(np.mean(test_ctr_loss_all), 5)
        test_ctcvr_loss = np.round(np.mean(test_ctcvr_loss_all), 5)
         
        result_list = [FLAGS.learning_rate, FLAGS.batch_size, test_ctr_auc, test_cvr_auc, \
                       test_ctcvr_auc, test_ctr_loss, test_ctcvr_loss, train_time, test_time]
 
        fmt_str = '{:<6}\t{:<6}\t{:<6}\t{:<6}\t{:<6}\t{:<6}\t{:<6}\t{}\t{}'
        header_row = ['eta', 'bs', 'ctr_auc', 'cvr_auc', 'ctcvr_auc', 'ctr_loss', 'ctcvr_loss', 'train_time', 'test_time']
        print(fmt_str.format(*header_row))
        print(fmt_str.format(*result_list))

def main(unused_argv):
    tf.reset_default_graph()

    # create dir
    base_path = '../tmp'
    if not os.path.exists(base_path):
        os.mkdir(base_path)

    # remove dir
    if os.path.isdir(FLAGS.model_saving_addr):
        shutil.rmtree(FLAGS.model_saving_addr)

    ###########################################################
    ###########################################################

    print('Loading data start!')
    tf.set_random_seed(123)

    train_values, test_values = read_data()

    print('Loading data done!')

    ########################################################################
    sparse_features = SparseFeatures(FLAGS.sparse_feature_list_fn)
    sparse_feature_list = sparse_features.get_feature_list()
    dense_features = DenseFeatures(FLAGS.dense_feature_list_fn)
    dense_feature_list = dense_features.get_feature_list()
    sparse_feature_columns = None
    dense_feature_columns = None
    sparse_csv_columns = sparse_features.get_csv_columns()
    sparse_feature_columns = sparse_features.get_feature_columns('train', embedding_dim=FLAGS.embedding_dim, steps_to_live=FLAGS.steps_to_live)
    dense_csv_columns = dense_features.get_csv_columns()

    if dense_csv_columns != "":
        csv_columns = "ctr_label,cvr_label,"+sparse_csv_columns+","+dense_csv_columns
        dense_feature_columns = dense_features.get_feature_columns()
    else:
        csv_columns = "ctr_label,cvr_label," + sparse_csv_columns
    print("csv_columns is: ", csv_columns)

    params = {
        'sparse_columns': sparse_feature_columns,
        'dense_columns' : dense_feature_columns
    }

    input_tensor_dict_train = func.parse_instance(train_values, FLAGS.batch_size, sparse_feature_list, dense_feature_list)
    input_tensor_dict_test = func.parse_instance(test_values, FLAGS.test_batch_size, sparse_feature_list, dense_feature_list)

    model = Model()
    ops_dict_train = get_ops(model, input_tensor_dict_train, params, 'train')
    ops_dict_test = get_ops(model, input_tensor_dict_test, params, 'test')
    train_test(ops_dict_train, ops_dict_test)

    for v in tf.trainable_variables():
        print(v)

if __name__ == '__main__':
  tf.app.run()

# # count # trainable params
# n_params = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
# print(n_params)
#
