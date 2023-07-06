#!/usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf

tf.app.flags.DEFINE_boolean("has_lineid", False, "whether the dataset has lineid")
tf.app.flags.DEFINE_string("model_saving_addr", "./tmp" , "")
tf.app.flags.DEFINE_integer("steps_print_train_info", 200, "")
tf.app.flags.DEFINE_boolean("save_model_ind", False, "whether to save model")
tf.app.flags.DEFINE_string("layer_dim_str", '512:256:128', "dim in hidden layers")
tf.app.flags.DEFINE_integer("num_hash_buckets", 200000, "number of hash buckets for features")
tf.app.flags.DEFINE_integer("batch_size", 64, "train batch size")
tf.app.flags.DEFINE_integer("test_batch_size", 64, "test batch size")
tf.app.flags.DEFINE_float("learning_rate", 0.1, "learning_rate")
tf.app.flags.DEFINE_integer("train_epoch", 2, "train epoch number")
tf.app.flags.DEFINE_integer("test_epoch", 1, "test epoch number")
tf.app.flags.DEFINE_integer("embedding_dim", 10, "dimension of embedding")
tf.app.flags.DEFINE_string("train_file_name", "./data/enc_merged_train_all_clk_esmm_single_valued.csv" , "")
tf.app.flags.DEFINE_string("test_file_name", "./data/enc_merged_test_all_clk_esmm_single_valued.csv" , "")

tf.app.flags.DEFINE_integer("steps_to_live", 8000000, "ev steps to live")
tf.app.flags.DEFINE_boolean("use_timeline", False, "use timeline")
tf.app.flags.DEFINE_boolean("tensor_fuse", False, "")
tf.app.flags.DEFINE_string("sparse_feature_list_fn", './data/sparse_ft_list_tb_single_valued.txt', "sparse feature list filename")
tf.app.flags.DEFINE_string("dense_feature_list_fn", './data/dense_ft_list.txt', "dense feature list filename")
tf.app.flags.DEFINE_string("graph_signature_name", 'predict_info',"")
tf.app.flags.DEFINE_string("model_output_name", 'model_output',"")
tf.app.flags.DEFINE_string("model_input_name", 'model_input',"")
tf.app.flags.DEFINE_integer("read_capacity", 1000, "TableRecordReader capacity, 0 for auto")
tf.app.flags.DEFINE_integer("read_threads", 24, "TableRecordReader num_threads, 0 for auto")

tf.app.flags.DEFINE_boolean("test_only", False, "")
tf.app.flags.DEFINE_string("optimizer", 'Adagrad',"")
tf.app.flags.DEFINE_float("adam_beta1", 0.9, "")
tf.app.flags.DEFINE_float("adam_beta2", 0.99, "")
tf.app.flags.DEFINE_float("adam_epsilon", 1e-8, "")
tf.app.flags.DEFINE_boolean("use_smooth_decay", False, "")
tf.app.flags.DEFINE_integer("decay_steps", 1000000, "")
tf.app.flags.DEFINE_float("decay_rate", 0.9, "")
tf.app.flags.DEFINE_integer("max_slot_num", 1000, "")

FLAGS = tf.app.flags.FLAGS

