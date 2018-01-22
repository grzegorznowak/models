import pandas as pd
import os
import random
import numpy as np

def get_shuffled_training_set(batch_size, verification_batches):
  X_train_batches, y_train_batches, X_verify_bacthes, y_verify_batches = get_data_batches(batch_size, verification_batches)
  train_batches = list(zip(X_train_batches, y_train_batches))
  random.shuffle(train_batches)
  verification_batches = list(zip(X_verify_bacthes, y_verify_batches))
  random.shuffle(verification_batches)


  X_train_batches_shuffled, y_train_batches_shuffled = zip(*train_batches)
  X_verify_batches_shuffled, y_verify_batches_shuffled = zip(*verification_batches)

  return X_train_batches_shuffled, y_train_batches_shuffled, X_verify_batches_shuffled, y_verify_batches_shuffled

def get_data_batches(batch_size, verification_batches):
  raw_data = pd.read_csv(os.path.join("/home/grzegorz/workspace/mql-csv-export/Data/", "mt4_EURUSD_1M_range_1_OHCLT_train_data.csv"))
  verification_size = verification_batches + batch_size + 20

  training_seq     = np.array(raw_data[verification_size:])
  verification_seq = np.array(raw_data[:verification_size])
  # split into items of input_size
  X_train_batches = np.array([[training_seq[i : batch_size + i]]
               for i in range(len(training_seq) - batch_size - 1)])
  y_train_batches  = np.array([[[training_seq[batch_size + i][0:4]]]
               for i in range(len(training_seq) - batch_size)])
  X_verify_batches = np.array([[verification_seq[i : batch_size + i]]
                     for i in range(len(verification_seq) - batch_size - 1)])
  y_verify_batches = np.array([[[verification_seq[batch_size + i][0:4]]]
                     for i in range(len(verification_seq) - batch_size)])

  return X_train_batches, y_train_batches, X_verify_batches, y_verify_batches
