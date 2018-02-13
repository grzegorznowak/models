import pandas as pd
import os
import random
import numpy as np
from numpy import genfromtxt
from functools import lru_cache


data_file = os.path.join("/mnt/c/Users/grzegorz/workspace/mql-csv-export/Data/", "test_mt4_EURUSD_1_range_1_UNnormalized_NOdatetime_OHCLT_train_data.csv")


# data_folder = os.path.join("/mnt/c/Users/grzegorz/AppData/Roaming/MetaQuotes/Terminal/3BD2B5E5A5264AFE17C1E2DDC7D6B381/tester/files/mt4_EURUSD_1__range_1_UNnormalized_NOdatetime_8-17_1-5_OHCLT_train_data")
data_folder              = os.path.join("/mnt/c/Users/grzegorz/workspace/mql-csv-export/Data/EURUSD_1_range_1_UNnormalized_datetime_volume_0-24_1-5_2015-2016_OHCLT_train_data")
verification_data_folder = os.path.join("/mnt/c/Users/grzegorz/AppData/Roaming/MetaQuotes/Terminal/3BD2B5E5A5264AFE17C1E2DDC7D6B381/tester/files/EURUSD_1_range_1_UNnormalized_datetime_volume_0-24_1-5_2017_OHCLT_train_data")

@lru_cache(maxsize=None)
def get_files_in_folder(directory):
  return os.listdir(os.fsencode(directory))


def get_shuffled_training_set(from_page, page_size, batch_size, verification_batches):
  return get_data_batches(from_page, page_size, batch_size, verification_batches)
  # dont shuffle for now, for easier insight into learning issues
  # X_train_batches, y_train_batches, X_verify_bacthes, y_verify_batches = get_data_batches(from_page, page_size, batch_size, verification_batches)
  # train_batches = list(zip(X_train_batches, y_train_batches))
  # random.shuffle(train_batches)
  # verification_batches = list(zip(X_verify_bacthes, y_verify_batches))
  # random.shuffle(verification_batches)
  #
  # X_train_batches_shuffled, y_train_batches_shuffled = zip(*train_batches)
  # X_verify_batches_shuffled, y_verify_batches_shuffled = zip(*verification_batches)
  #
  # return X_train_batches_shuffled, y_train_batches_shuffled, X_verify_batches_shuffled, y_verify_batches_shuffled

def get_data_batches(from_page, page_size, batch_size, verification_batches):
  batch_from = from_page * page_size
  raw_data = pd.read_csv(data_file, skiprows=batch_from, nrows=page_size)
  verification_size = verification_batches + batch_size

  training_seq     = np.array(raw_data[verification_size:])
  verification_seq = np.array(raw_data[:verification_size])
  # split into items of input_size
  X_train_batches = np.array([[training_seq[i : batch_size + i]]
               for i in range(len(training_seq) - batch_size - 1)])

  y_train_batches  = np.array([[training_seq[i + 1 : batch_size + i +1]]
               for i in range(len(training_seq) - batch_size)])

  X_verify_batches = np.array([[verification_seq[i : batch_size + i]]
                     for i in range(len(verification_seq) - batch_size - 1)])

  y_verify_batches = np.array([[[verification_seq[batch_size + i][1:4]]]
                     for i in range(len(verification_seq) - batch_size)])

  return X_train_batches, y_train_batches, X_verify_batches, y_verify_batches

def get_data_from_file(filename, batch_size):
  raw_data          = pd.read_csv(filename)
  training_seq      = np.array(raw_data)

  # split into items of input_size
  X_train_batches = np.array([[training_seq[i : batch_size + i]]
                              for i in range(0, len(training_seq) - batch_size - 1, batch_size)])

  y_train_batches = np.array([[list(map(lambda value: [value[1]], training_seq[i + 1 : batch_size + i + 1]))]  # just H bars as a prediction pls (no close, just to see if it helps)
                              for i in range(0, len(training_seq) - batch_size, batch_size)])


  return X_train_batches, y_train_batches


def get_steps_from_file(file, time_steps, directory):
  filename = os.fsdecode(file)
  filepath = os.path.join(directory, filename)

  seq = genfromtxt(filepath, delimiter=',')

  # split into items of input_size
  X_train_steps = [[seq[i : time_steps + i]]
                            for i in range(0, len(seq) - time_steps - 1, time_steps)]

  y_train_steps = [[list(map(lambda value: [value[1]], seq[i + 1 : time_steps + i + 1]))]  # just H bars as a prediction pls (no close, just to see if it helps)
                            for i in range(0, len(seq) - time_steps, time_steps)]

  return X_train_steps, y_train_steps

"""
creates parallel batches from a couple of files.
batch_size defines how many files to take into constructing data
"""
def get_data_batch_from_files(batch_size):
  return []

def parse_cmdline(ars):
  is_training     = (ars[1] == "train")
  is_continue     = (ars[1] == "continue")
  restore_name    = ""
  start_day_input = ""
  end_day_input   = ""

  if is_continue:
    try:
      start_day_input = ars[3]
    except IndexError:
      start_day_input = 0

    try:
      end_day_input   = ars[4]
    except IndexError:
      end_day_input   = 0
    restore_name    = ars[2]
  else:
    try:
      start_day_input = ars[2]
    except IndexError:
      start_day_input = 0

    try:
      end_day_input   = ars[3]
    except IndexError:
      end_day_input   = 0

  return is_training, is_continue, restore_name, start_day_input, end_day_input


def get_total_data_batches_count_in_train_folder(batch_size):
  return len(get_files_in_folder(data_folder)) // batch_size - 1  # last batch will be foobar, drop it

def get_total_data_batches_count_in_verify_folder(batch_size):
  return len(get_files_in_folder(verification_data_folder)) // batch_size - 1

@lru_cache(maxsize=2048)
def get_data_batches_from_folder(folder, index, batch_size, steps_count):
  files       = get_files_in_folder(folder)
  batch_files = files[index * batch_size : index * batch_size + batch_size]
  X_batches = []
  y_batches = []

  for file in batch_files:
    X_batch, y_batch = get_steps_from_file(file, steps_count, folder)
    X_batches.append(X_batch)
    y_batches.append(y_batch)


  shortest_batch_len = np.amin([len(l) for l in X_batches])
  trimmed_Xs = [original[0:shortest_batch_len] for original in X_batches]
  trimmed_ys = [original[0:shortest_batch_len] for original in y_batches]

  return np.concatenate( trimmed_Xs, axis=1 ), np.concatenate( trimmed_ys, axis=1 )

def get_verification_data_bathes(index, batch_size, steps_count):
  return get_data_batches_from_folder(verification_data_folder, index, batch_size, steps_count)


def get_train_data_batches(index, batch_size, steps_count):
  return get_data_batches_from_folder(data_folder, index, batch_size, steps_count)


def get_train_data_batch_from_folder(index, batch_size):
  return get_data_batch_from_folder(index, batch_size, data_folder)


def get_verify_data_batch_from_folder(index, batch_size):
  return get_data_batch_from_folder(index, batch_size, verification_data_folder)

def get_random_data_batch_from_folder(batch_size):
  files        = get_files_in_folder(data_folder)
  random_index = random.randint(0, len(files) -1)

  return get_data_batch_from_folder(random_index, batch_size, data_folder), random_index

def get_random_data_batch_from_verification_folder(batch_size):
  files        = get_files_in_folder(verification_data_folder)
  random_index = random.randint(0, len(files) -1)
  return get_data_batch_from_folder(random_index, batch_size, verification_data_folder), random_index

@lru_cache(maxsize=2048)
def get_data_batch_from_folder(batch_index, batch_size, directory):
  files     = get_files_in_folder(directory)
  filename  = os.fsdecode(files[batch_index])

  return get_data_from_file(os.path.join(directory, filename), batch_size)


def data_row_count():
  with open(data_file) as f:
    for i, l in enumerate(f):
      pass
  return i + 1


# how many data
def data_batches_count(row_count, batch_size):
  return max(1, row_count // (1000000 // batch_size)), 1000000 // batch_size


def signal_stats(predictions, labels, threshold):
  negative_threshold = -1 * threshold   # also count outliers with reversed sign
  total = 0
  found = 0
  wrong = 0

  response = np.asarray(predictions)
  y_val    = np.asarray(labels)

  it = np.nditer(y_val, flags=['multi_index'])
  while not it.finished:
    itm = it[0]
    if itm <= negative_threshold:
      total += 1
      if response[it.multi_index] <= negative_threshold:
        found += 1
    if itm >= threshold:
      total += 1
      if response[it.multi_index] >= threshold:
        found += 1
    it.iternext()

  it = np.nditer(response, flags=['multi_index'])
  while not it.finished:
    itm = it[0]
    if itm <= negative_threshold:
      if y_val[it.multi_index] > negative_threshold:
        wrong += 1
    if itm >= threshold:
      if y_val[it.multi_index] < negative_threshold:
        wrong += 1
    it.iternext()

  return total, found, wrong
