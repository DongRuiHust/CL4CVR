#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import time
import sys
import numpy as np
from flag_define import *
import tensorflow as tf

# load sparse_features and build feature_list, csv_columns, sparse_column_list
class SparseFeatures:
  def __init__(self, features_spec_file):
    self.features_spec_file = features_spec_file
    # list of (name, slot)
    self.feature_list = []

    if not os.path.exists(self.features_spec_file):
      raise ValueError('sparse features_spec_file not detected.')

    column_name_list = []
    f = open(self.features_spec_file, 'r')
    for line in f.readlines():
      if line.startswith('#'):
        continue
      fields = line.split('\t')
      if len(fields) == 2:
        fea_name, fea_slot = fields[0].strip(), fields[1].strip()
      else:
        raise ValueError('features_spec_file wrong line detected: %s' % line)
      # odps table use slot as column name
      column_name_list.append("slot_"+fea_slot)

      self.feature_list.append((fea_name, fea_slot))
    self.csv_columns = ",".join(column_name_list)
    print("load feature_spec %s, with %d sparse feature, csv_columns=%s" % (self.features_spec_file, len(self.feature_list), self.csv_columns))

  def get_feature_list(self):
    return self.feature_list

  def get_csv_columns(self):
    return self.csv_columns

  # gen feature_column_list with feature_list conf
  # params:
  #   @embedding_dim: embedding dim for one slot
  
  def get_feature_columns(self, mode, embedding_dim = 10, steps_to_live = 200000):
    if mode == "train":
      initializer = tf.truncated_normal_initializer(mean = 0.0, stddev = 0.01)
    else:
      initializer = tf.zeros_initializer()
    sparse_column_list = []
    sparse_feature_len = len(self.feature_list)

    sparse_id_cols = []
    for i in range(sparse_feature_len):
      fea_name = self.feature_list[i][0]
      fea_slot = self.feature_list[i][1]
      print("sparse fea %s:%s" % (fea_name, fea_slot))
      sparse_id_column = tf.feature_column.categorical_column_with_hash_bucket(str(FLAGS.max_slot_num+i), hash_bucket_size=20000, dtype=tf.string)
          
      # sparse_embedding_column = tf.feature_column.embedding_column(sparse_id_column, embedding_dim, combiner = 'sum', initializer = initializer)
      # sparse_column_list.append(sparse_embedding_column)
      
      sparse_id_cols.append(sparse_id_column)
    
    sparse_column_list = tf.feature_column.shared_embedding_columns(sparse_id_cols, embedding_dim, combiner = 'sum', initializer = initializer)
    return sparse_column_list


#load dense features and build feature_list, csv_columns, dense_column_list
class DenseFeatures:
  def __init__(self, features_spec_file):
    self.features_spec_file = features_spec_file
    # list of (name, dim)
    self.feature_list = []
    self.csv_columns = ""
    if not os.path.exists(self.features_spec_file):
      print ('dense features_spec_file not detected.')
      return
    column_name_list = []
    f = open(self.features_spec_file, 'r')
    for line in f.readlines():
      if line.startswith('#'):
        continue
      # name dim
      fields = line.split('\t')
      if len(fields) == 2:
        fea_name, fea_dim = fields[0].strip(), fields[1].strip()
      else:
        raise ValueError('features_spec_file wrong line detected: %s' % line)
      # odps table use name as column name
      column_name_list.append(fea_name)    
      self.feature_list.append((fea_name, fea_dim))
    self.csv_columns = ",".join(column_name_list)
    print("load feature_spec %s, with %d dense feature, csv_columns=%s" % (features_spec_file, len(self.feature_list), self.csv_columns))

  def get_feature_list(self):
    return self.feature_list

  def get_csv_columns(self):
    return self.csv_columns

  # gen dense feature_column_dict with feature_list conf
  def get_feature_columns(self):
    dense_column_list = []
    dense_feature_len = len(self.feature_list)

    for i in range(dense_feature_len):
      fea_name = self.feature_list[i][0]
      fea_dim = int(self.feature_list[i][1])
      print("dense fea %s, with dim=%d" % (fea_name, fea_dim))
      dense_column = tf.contrib.layers.real_valued_column(column_name = str(FLAGS.max_slot_num*2+i), dimension = fea_dim, default_value = 0.0, dtype=tf.float32)
      dense_column_list.append(dense_column)
    return dense_column_list

