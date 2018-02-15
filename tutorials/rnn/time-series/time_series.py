import gc
import sys
from tqdm import tqdm
import numpy as np
import tensorflow as tf
import time_series_data as tsd

from random   import randint
from datetime import datetime
from pympler import tracker
from pympler import summary
from pympler import muppy
from functools import lru_cache

tr = tracker.SummaryTracker()

class LaptopCPUConfig(object):
  name           = "LaptopCPU"
  rnn_neurons    = 300
  batch_size     = 1
  rnn_layers     = 1
  n_outputs      = 4
  n_inputs       = 4
  initial_lr     = 0.001   #initial learning rate
  decay_lr       = 0.9
  keep_prob      = 0.5     # dropout only on RNN layer(s)
  full_mse_count = 2
  def create_rnn(self):
    return tf.contrib.rnn.BasicRNNCell(num_units=self.rnn_neurons) # try using faster cells

class DesktopCPUConfig(object):
  name           = "DesktopCPU"
  rnn_neurons    = 500
  batch_size     = 6
  rnn_layers     = 2
  n_outputs      = 4
  n_inputs       = 4
  initial_lr     = 0.0005   #initial learning rate
  decay_lr       = 0.99
  keep_prob      = 0.99     # dropout only on RNN layer(s)
  full_mse_count = 10

  def create_rnn(self):
    return tf.contrib.rnn.GRUCell(num_units=self.rnn_neurons) #tf.nn.relu , use_peepholes=True

class RNN_50_1_5(object):
    name           = "RNN_50_1_5"
    rnn_neurons    = 50
    batch_size     = 5
    rnn_layers     = 1
    n_inputs       = 6
    n_outputs      = 1
    initial_lr     = 0.0005   #initial learning rate
    decay_lr       = 0.95
    keep_prob      = 0.5     # dropout only on RNN layer(s)
    full_mse_count = 20

    def create_rnn(self):
        return tf.contrib.rnn.BasicRNNCell(num_units=self.rnn_neurons) # try using faster cells?

class RNN_300_3_5(object):
    name           = "RNN_300_2_5"
    rnn_neurons    = 300
    batch_size     = 5
    rnn_layers     = 3
    n_inputs       = 6
    n_outputs      = 1
    initial_lr     = 0.0005   #initial learning rate
    decay_lr       = 0.95
    keep_prob      = 0.5     # dropout only on RNN layer(s)
    full_mse_count = 2

    def create_rnn(self):
        return tf.contrib.rnn.BasicRNNCell(num_units=self.rnn_neurons) # try using faster cells?


class GRU_300_2_5(object):
    name           = "GRU_300_2_5"
    rnn_neurons    = 300
    batch_size     = 5
    steps_number   = 1
    rnn_layers     = 2
    n_inputs       = 6
    n_outputs      = 1
    initial_lr     = 0.001   #initial learning rate
    decay_lr       = 0.95
    keep_prob      = 0.5     # dropout only on RNN layer(s)
    full_mse_count = 2

    def create_rnn(self):
        return tf.contrib.rnn.GRUCell(num_units=self.rnn_neurons) # try using faster cells?

class GRU_300_2_1(object):
    name           = "GRU_300_2_1"
    rnn_neurons    = 300
    batch_size     = 1
    rnn_layers     = 2
    n_inputs       = 6
    n_outputs      = 1
    initial_lr     = 0.001   #initial learning rate
    decay_lr       = 0.97
    keep_prob      = 0.5     # dropout only on RNN layer(s)
    full_mse_count = 2

    def create_rnn(self):
        return tf.contrib.rnn.GRUCell(num_units=self.rnn_neurons) # try using faster cells?

class GRU_300_2_10_5(object):
    name           = "GRU_300_2_10_5"
    rnn_neurons    = 300
    batch_size     = 10
    steps_number   = 5
    rnn_layers     = 2
    n_inputs       = 6
    n_outputs      = 1
    initial_lr     = 0.001   #initial learning rate
    decay_lr       = 0.98
    keep_prob      = 0.5     # dropout only on RNN layer(s)
    full_mse_count = 4

    def create_rnn(self):
        return tf.contrib.rnn.GRUCell(num_units=self.rnn_neurons) # try using faster cells?


class GRU_400_3_10_1(object):
    name           = "GRU_400_3_10_1"
    rnn_neurons    = 400
    batch_size     = 10
    steps_number   = 1
    rnn_layers     = 3
    n_inputs       = 6
    n_outputs      = 1
    initial_lr     = 0.0005   #initial learning rate
    decay_lr       = 0.99
    keep_prob      = 0.5     # dropout only on RNN layer(s)
    full_mse_count = 3

    def create_rnn(self):
        return tf.contrib.rnn.GRUCell(num_units=self.rnn_neurons) # try using faster cells?


class GRU_300_2_10_10_SGD(object):
    name           = "GRU_300_2_10_10_SGD"
    rnn_neurons    = 300
    batch_size     = 10
    steps_number   = 10
    rnn_layers     = 2
    n_inputs       = 6
    n_outputs      = 1
    initial_lr     = 0.002   #initial learning rate
    ramp_start     = 0.02 * initial_lr
    ramp_steps     = 10
    ramp_increase  = (initial_lr - ramp_start) / ramp_steps
    decay_lr       = 0.5
    decay_step     = 10  # how many epochs to process before decreasing LR
    keep_prob      = 0.5     # dropout only on RNN layer(s)
    full_mse_count = 3

    def create_rnn(self):
        return tf.contrib.rnn.GRUCell(num_units=self.rnn_neurons) # try using faster cells?

    def get_optimizer(self, learning_rate):
        return tf.train.GradientDescentOptimizer(learning_rate=learning_rate)


class GRU_300_5_10_20_SDG(object):
    name           = "GRU_300_5_10_20_SDG"
    rnn_neurons    = 300
    rnn_layers     = 5
    batch_size     = 10
    steps_number   = 20
    n_inputs       = 6
    n_outputs      = 1
    initial_lr     = 0.001 * batch_size * steps_number    #initial learning rate
    ramp_start     = 0.1 * initial_lr
    ramp_steps     = 10
    ramp_increase  = (initial_lr - ramp_start) / ramp_steps
    decay_lr       = 0.1
    decay_step     = 30  # how many epochs to process before decreasing LR
    keep_prob      = 0.5     # dropout only on RNN layer(s)
    full_mse_count = 2

    def create_rnn(self):
        return tf.contrib.rnn.GRUCell(num_units=self.rnn_neurons) # try using faster cells?

    def get_optimizer(self, learning_rate):
        return tf.train.GradientDescentOptimizer(learning_rate=learning_rate)

class GRU_400_2_10_20_SDG(object):
    name           = "GRU_400_2_10_20_SDG"
    rnn_neurons    = 400
    rnn_layers     = 2
    batch_size     = 10
    steps_number   = 20
    n_inputs       = 6
    n_outputs      = 1
    initial_lr     = 0.001 * batch_size * steps_number * n_inputs   #initial learning rate
    ramp_start     = 0.02 * initial_lr
    ramp_steps     = 10
    ramp_increase  = (initial_lr - ramp_start) / ramp_steps
    decay_lr       = 0.5
    decay_step     = 20  # how many epochs to process before decreasing LR
    keep_prob      = 0.5     # dropout only on RNN layer(s)
    full_mse_count = 2

    def create_rnn(self):
        return tf.contrib.rnn.GRUCell(num_units=self.rnn_neurons) # try using faster cells?

    def get_optimizer(self, learning_rate):
        return tf.train.GradientDescentOptimizer(learning_rate=learning_rate)

model_config = GRU_400_2_10_20_SDG

class GraphWrapper():
  def __init__(self, graph, init_op, initial_state_placeholder, multi_layer_cell,
               final_state_op, train_day, X, y, epoch, loss, outputs_int, outputs, learning_rate,
               training_op, keep_prob, summary_op, rnn_outputs, final_state, stacked_rnn_outputs,
               stacked_outputs, is_training, saver, totals_summary_op, total_train_mse,
               total_verification_mse, mild_signals_found, mild_signals_wrong,
               strong_signals_found, strong_signals_wrong):
     self.graph                     = graph
     self.init_op                   = init_op
     self.initial_state_placeholder = initial_state_placeholder
     self.multi_layer_cell          = multi_layer_cell
     self.final_state_op            = final_state_op
     self.train_day_placeholder     = train_day
     self.X_placeholder             = X
     self.y_placeholder             = y
     self.epoch_placeholder         = epoch
     self.loss_placeholder          = loss
     self.outputs_placeholder       = outputs_int
     self.outputs_raw_placeholder   = outputs
     self.learning_rate_placeholder = learning_rate
     self.training_op_placeholder   = training_op
     self.keep_prob_placeholder     = keep_prob
     self.summary_op                = summary_op
     self.rnn_outputs               = rnn_outputs
     self.final_state               = final_state
     self.stacked_rnn_outputs       = stacked_rnn_outputs
     self.stacked_outputs           = stacked_outputs
     self.is_training_placeholder   = is_training
     self.saver                     = saver
     self.totals_summary_op         = totals_summary_op
     self.total_train_mse           = total_train_mse
     self.total_verification_mse    = total_verification_mse
     self.mild_signals_found        = mild_signals_found
     self.mild_signals_wrong        = mild_signals_wrong
     self.strong_signals_found      = strong_signals_found
     self.strong_signals_wrong      = strong_signals_wrong


def build_rnn_time_series_graph(graph_config):

  create_dropout = lambda cell: tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=keep_prob)

  graph = tf.Graph()
  with graph.as_default():
    keep_prob      = tf.placeholder(tf.float32, None, name="keep_prob")
 #   he_init        = tf.contrib.layers.variance_scaling_initializer()

    X              = tf.placeholder(tf.float32, [graph_config.batch_size, graph_config.steps_number, graph_config.n_inputs] , name="X")
    y              = tf.placeholder(tf.float32, [graph_config.batch_size, graph_config.steps_number, graph_config.n_outputs], name="y")
    learning_rate  = tf.placeholder(tf.float32, None, name="learning_rate")
    epoch          = tf.placeholder(tf.int16  , name="epoch")
    train_day      = tf.placeholder(tf.int16  , name="train_day")
    is_training    = tf.placeholder_with_default(True  , shape=(), name="is_training")

    total_train_mse        = tf.placeholder(tf.float32)
    total_verification_mse = tf.placeholder(tf.float32)
    mild_signals_found     = tf.placeholder(tf.float32)
    mild_signals_wrong     = tf.placeholder(tf.float32)
    strong_signals_found   = tf.placeholder(tf.float32)
    strong_signals_wrong   = tf.placeholder(tf.float32)

    initial_state_placeholder = tf.placeholder(tf.float32, [graph_config.batch_size, graph_config.rnn_neurons * graph_config.rnn_layers], name="initial_state_placeholder")

    cell_layers    = [graph_config.create_rnn()  for _ in range(graph_config.rnn_layers)]
    dropout_layers = [create_dropout(cell_layer) for cell_layer in cell_layers]
    #
    # hidden_input_0             = tf.layers.dense(X, graph_config.n_inputs, name="hidden_input_0")
    # hidden_normalization_0     = tf.contrib.layers.layer_norm(hidden_input_0, begin_norm_axis=2)  # normalize each batch separately
    # hidden_normalization_0_act = tf.nn.elu(hidden_normalization_0)

    multi_layer_cell           = tf.contrib.rnn.MultiRNNCell(dropout_layers, state_is_tuple=False)
    rnn_outputs, final_state   = tf.nn.dynamic_rnn(multi_layer_cell, X,
                                                   dtype=tf.float32,
                                                   initial_state=initial_state_placeholder)
    print(rnn_outputs)
    stacked_rnn_outputs = tf.reshape(rnn_outputs, [-1, graph_config.rnn_neurons], name="stacked_rnn_outputs")
    stacked_outputs     = tf.layers.dense(stacked_rnn_outputs, graph_config.n_outputs, name="stacked_outputs")
    outputs             = tf.reshape(stacked_outputs,
                                     [graph_config.batch_size, graph_config.steps_number, graph_config.n_outputs],
                                     name="outputs")

    loss        = tf.reduce_mean(tf.abs(y - outputs))
 #   optimizer   = tf.train.AdamOptimizer(learning_rate=learning_rate)
    optimizer  = graph_config.get_optimizer(learning_rate)

 #   gvs        = optimizer.compute_gradients(loss)
  #  capped_gvs = [(tf.clip_by_value(grad, -20., 20.), var) for grad, var in gvs]
 #   training_op = optimizer.apply_gradients(gvs, name="training_op")

    training_op    = optimizer.minimize(loss, name="training_op")

    outputs_int = tf.cast(outputs, tf.int32)

    # mergable summaries
    learning_rate_summary = tf.summary.scalar("learning_rate", learning_rate)
    epoch_summary         = tf.summary.scalar("epoch"        , epoch)
    train_day_summary     = tf.summary.histogram("train_day"    , train_day)
    mse_summary_summary   = tf.summary.scalar("mse_summary"  , loss)

    weights_output_summary = tf.summary.histogram("weights_output", outputs)
    new_states_summary     = tf.summary.histogram("new_states", final_state)

    init = tf.global_variables_initializer()

    summary_op = tf.summary.merge([learning_rate_summary, epoch_summary, train_day_summary, mse_summary_summary,
                                   weights_output_summary, new_states_summary])

    # specific summaries
    total_train_mse_summary         = tf.summary.scalar("total_train_mse"  , total_train_mse)
    total_verification_mse_summary  = tf.summary.scalar("total_verification_mse"  , total_verification_mse)
    mild_signals_found_summary      = tf.summary.scalar("mild_signals_found_summary (correct / total) max == 1 the higher the number the more accurate the network is"  , mild_signals_found)
    mild_signals_wrong_summary      = tf.summary.scalar("mild_signals_wrong_summary (false positives / correct) the lower the number the better"  , mild_signals_wrong)
    strong_signals_found_summary    = tf.summary.scalar("strong_signals_found_summary (correct / total) max == 1 the higher the number the more accurate the network is"  , strong_signals_found)
    strong_signals_wrong_summary    = tf.summary.scalar("strong_signals_wrong_summary (false positives / correct) the lower the number the better"  , strong_signals_wrong)

    totals_summary_op = tf.summary.merge([total_train_mse_summary, total_verification_mse_summary,
                                          mild_signals_found_summary, mild_signals_wrong_summary,
                                          strong_signals_found_summary, strong_signals_wrong_summary])

    saver = tf.train.Saver(max_to_keep=0)

  return GraphWrapper(graph, init, initial_state_placeholder, multi_layer_cell, final_state,
                      train_day, X, y, epoch, loss, outputs_int, outputs, learning_rate, training_op,
                      keep_prob, summary_op, rnn_outputs, final_state, stacked_rnn_outputs,
                      stacked_outputs, is_training, saver, totals_summary_op,
                      total_train_mse, total_verification_mse, mild_signals_found, mild_signals_wrong,
                      strong_signals_found, strong_signals_wrong)


def measure_performance(zero_state, X, y, graph_wrapper, sess, verification_day_index):

  print("\tMeasuring performance based on a sample from the day of index ", verification_day_index, "\n")

  X_placeholder = graph_wrapper.X_placeholder
  y_placeholder = graph_wrapper.y_placeholder
  outputs       = graph_wrapper.outputs_placeholder
  loss          = graph_wrapper.loss_placeholder
  keep_prob     = graph_wrapper.keep_prob_placeholder
  new_state_op  = graph_wrapper.final_state_op
  initial_state_placeholder = graph_wrapper.initial_state_placeholder
  is_training_placeholder = graph_wrapper.is_training_placeholder

  next_state = zero_state
  sum_mse    = 0
  for batch_number in range(len(X)):
    mse, next_state, train_response = sess.run([loss, new_state_op, outputs],
                                               feed_dict={X_placeholder: X[batch_number],
                                                          y_placeholder: y[batch_number],
                                                          keep_prob: 1,
                                                          initial_state_placeholder: next_state,
                                                          is_training_placeholder:False})
    sum_mse += mse

    if batch_number % 20 == 0: # print out details only every 10
      print("Verification batch number: ", batch_number,
          "\tvY vs vResp: \n",
            np.transpose(y[batch_number]), "<->\n",  np.transpose(train_response),
          "\tBatch MSE: ", mse)


  print("\nTOTAL Average Verfication MSE: \t", sum_mse / len(X), "\n")


def training_iteration(previous_state, current_learning_rate, iteration, epoch, random_index, train_X, train_y, graph_wrapper, session, saver, save_dir, file_writer, training_config):
  initial_state_placeholder = graph_wrapper.initial_state_placeholder
  train_day_placeholder     = graph_wrapper.train_day_placeholder
  new_state_op  = graph_wrapper.final_state_op
  X             = graph_wrapper.X_placeholder
  y             = graph_wrapper.y_placeholder
  epoch_holder  = graph_wrapper.epoch_placeholder
  keep_prob     = graph_wrapper.keep_prob_placeholder
  learning_rate = graph_wrapper.learning_rate_placeholder
  training_op   = graph_wrapper.training_op_placeholder
  outputs       = graph_wrapper.outputs_placeholder
  loss          = graph_wrapper.loss_placeholder
  is_training_placeholder = graph_wrapper.is_training_placeholder

  train, new_state = session.run([training_op, new_state_op],
                                      feed_dict={X: train_X, y: train_y, keep_prob: training_config.keep_prob,
                                                 learning_rate: current_learning_rate,
                                                 initial_state_placeholder: previous_state})


  if iteration % 2000 == 0:
    summary  = session.run(graph_wrapper.summary_op,
                          feed_dict={
                             X: train_X, y: train_y, keep_prob: 1,
                             initial_state_placeholder: previous_state, learning_rate: current_learning_rate,
                             epoch_holder: epoch,
                             train_day_placeholder: random_index,
                             is_training_placeholder: False})

    file_writer.add_summary(summary, iteration)

  if iteration % 5000 == 0:
    train_response, mse = session.run([outputs, loss],
                                 feed_dict={X: train_X, y: train_y, keep_prob: 1,
                                            initial_state_placeholder: previous_state,
                                            is_training_placeholder: False})

    print("epoch: ", epoch, ", (ticks) iteration: ", iteration)
    plain_y        = np.transpose(train_y)
    plain_response = np.transpose(train_response)
    print("last train_y vs output: \n"  , plain_y, "\t -> \n", plain_response, "\tMSE:"  , mse)
    print("current LR: ", current_learning_rate)



    # print(np.argwhere(plain_response > -2))


  if iteration % 150000 == 0:  # save network rarily
    saver.save(session, save_dir + "model_" + str(iteration) + "_" + str(epoch) + ".ckpt")

  return new_state

def get_stats_for_data_set(zero_state, training_config, graph_wrapper, sess, days_worth_of_data, data_getter):

    total_mse     = 0.0
    mild_signals_found   = 0
    strong_signals_found = 0
    mild_signals_total   = 0
    strong_signals_total = 0
    mild_signals_wrong   = 0
    strong_signals_wrong = 0

    mild_threshold   = 10
    strong_threshold = 25

    new_state_op  = graph_wrapper.final_state_op
    X             = graph_wrapper.X_placeholder
    y             = graph_wrapper.y_placeholder
    loss          = graph_wrapper.loss_placeholder
    keep_prob     = graph_wrapper.keep_prob_placeholder
    outputs       = graph_wrapper.outputs_placeholder
    initial_state_placeholder  = graph_wrapper.initial_state_placeholder

    for data_batch in tqdm(range(days_worth_of_data)):
        X_val, y_val    = data_getter(data_batch, training_config.batch_size, training_config.steps_number)
        data_batch_size = len(X_val)

        previous_state_value = zero_state()
        total_day_mse = 0.0

        for data_iteration in range(data_batch_size):
            response, previous_state_value, mse = sess.run([outputs, new_state_op, loss],
                                                feed_dict={X: X_val[data_iteration], y: y_val[data_iteration],
                                                           keep_prob: 1,
                                                           initial_state_placeholder: previous_state_value})


            mild_signals_total_in_iteration, mild_signals_found_in_iteration, mild_signals_wrong_in_teration       = tsd.signal_stats(response, y_val[data_iteration], mild_threshold)
            strong_signals_total_in_iteration, strong_signals_found_in_iteration, strong_signals_wrong_in_teration = tsd.signal_stats(response, y_val[data_iteration], strong_threshold)

            mild_signals_total += mild_signals_total_in_iteration
            mild_signals_found += mild_signals_found_in_iteration
            mild_signals_wrong += mild_signals_wrong_in_teration

            strong_signals_total += strong_signals_total_in_iteration
            strong_signals_found += strong_signals_found_in_iteration
            strong_signals_wrong += strong_signals_wrong_in_teration

            total_day_mse += mse
        total_mse += total_day_mse / data_batch_size

    return (total_mse / days_worth_of_data), (mild_signals_found / mild_signals_total), (mild_signals_wrong / mild_signals_total), (strong_signals_found / strong_signals_total), (strong_signals_wrong / strong_signals_total)

def main(_):

  is_training, is_continue, restore_name, start_day_input, end_day_input = tsd.parse_cmdline(sys.argv)

  if(len(sys.argv) < 1):
    print("wrong usage")
    os.exit(1)

  training_config  = model_config()

  # @TODO: need to redo those CMD params logic when they grow in number. Just stick to the bruteforce IF power
  if is_training or is_continue:

    epochs           = 500
    graph_wrapper    = build_rnn_time_series_graph(training_config)
    init_op          = graph_wrapper.init_op
    saver            = graph_wrapper.saver
    training_session = tf.Session(graph=graph_wrapper.graph)


    with training_session as sess:

      train_data_batches_count  = tsd.get_total_data_batches_count_in_train_folder(training_config.batch_size)
      verify_data_batches_count = tsd.get_total_data_batches_count_in_verify_folder(training_config.batch_size)

      # it doesn't mean that much anymore, but is a good heuristic to skip to another epoch after
      # train_data_batches_count worth of samples has passed by
      end_day            = train_data_batches_count
      totals_summary_op  = graph_wrapper.totals_summary_op
      mild_signals_found   = graph_wrapper.mild_signals_found
      mild_signals_wrong   = graph_wrapper.mild_signals_wrong
      strong_signals_found = graph_wrapper.strong_signals_found
      strong_signals_wrong = graph_wrapper.strong_signals_wrong

      total_train_mse_op       = graph_wrapper.total_train_mse
      total_verification_mse_op = graph_wrapper.total_verification_mse

      if is_continue:
        print(restore_name)
        saver.restore(sess, restore_name)

      else:
        init_op.run()

      print("train_data_batches_count in folder", train_data_batches_count)

    #  @lru_cache(maxsize=2)
      def zero_state():
        #return np.zeros((1,training_config.rnn_neurons * training_config.rnn_layers))
        return np.random.rand(training_config.batch_size, training_config.rnn_neurons * training_config.rnn_layers)

      start_day = int(start_day_input)
      if int(end_day_input):
        end_day =  int(end_day_input)

      days_worth_of_data        = end_day - start_day
      days_between_mse_snapshot = days_worth_of_data // training_config.full_mse_count  # make full train/ver sets MSE snapshot every that many days
      for epoch in range(0, epochs):
        log_dir     = "{}/run-{}-{}-{}/".format('/tmp/time_series_logdir', datetime.utcnow().strftime("%Y%m%d%H%M%S"), epoch, training_config.name)
        save_dir    = "{}/run-{}-{}-{}/".format('/tmp/time_series', datetime.utcnow().strftime("%Y%m%d%H%M%S"), epoch, training_config.name)
        file_writer = tf.summary.FileWriter(log_dir, tf.get_default_graph())  # TODO: each epoch will create a redundant node in graph, make it more TF way

        epoch_iteration       = 0
        decay_multiplier      = -1
        if epoch <= training_config.ramp_steps:
            current_learning_rate = training_config.ramp_start + training_config.ramp_increase * epoch
        else:
            decay_multiplier = (epoch - training_config.ramp_steps) // training_config.decay_step
            current_learning_rate = training_config.initial_lr * (training_config.decay_lr**decay_multiplier)

        verification_X, verification_y = tsd.get_verification_data_bathes(0, training_config.batch_size, training_config.steps_number, randomize=True)
        print("days_between_mse_snapshot: ", days_between_mse_snapshot)

        for data_batch in range(days_worth_of_data):
          train_X, train_y = tsd.get_train_data_batches(data_batch, training_config.batch_size, training_config.steps_number, randomize=True)
          data_batch_size  = len(train_X)

          previous_state_value = zero_state()

          print("Training ", epoch, "epoch, \tlearning rate:", current_learning_rate,"\ttrain set len of: ", data_batch_size,
                " iterations, \ncurrent data batch: ",data_batch, ' / ', train_data_batches_count, ' simulating day no. ', data_batch,
                "decay_multiplier: ", decay_multiplier)

          if data_batch % 10 == 0:
            # print out verification stats every a few days worth of training
            measure_performance(zero_state(), verification_X, verification_y, graph_wrapper, sess, days_worth_of_data // 2)

          if data_batch % days_between_mse_snapshot == 0:
               total_verify_mse, mild_right_ratio, mild_wrong_ratio, strong_right_ratio, strong_wrong_ratio = get_stats_for_data_set(zero_state, training_config, graph_wrapper, sess, verify_data_batches_count, tsd.get_verification_data_bathes)
               # only 10% of train data please, just so we have some data but not clog the learning process
               total_train_mse, _, _, _, _  = get_stats_for_data_set(zero_state, training_config, graph_wrapper, sess, train_data_batches_count // 10 , tsd.get_train_data_batches)
               summary          = sess.run(totals_summary_op,
                                           feed_dict={total_train_mse_op: total_train_mse,
                                                      mild_signals_found: mild_right_ratio,
                                                      mild_signals_wrong: mild_wrong_ratio,
                                                      strong_signals_found: strong_right_ratio,
                                                      strong_signals_wrong: strong_wrong_ratio,
                                                      total_verification_mse_op: total_verify_mse})

               file_writer.add_summary(summary, data_batch)

          for data_iteration in range(data_batch_size):
            previous_state_value = training_iteration(previous_state_value, current_learning_rate, epoch_iteration, epoch, data_batch, train_X[data_iteration % data_batch_size], train_y[data_iteration % data_batch_size], graph_wrapper, sess, saver, save_dir, file_writer, training_config)
            epoch_iteration     += training_config.batch_size

        file_writer.close()
        saver.save(sess, save_dir + "model_final_" + str(epoch) + ".ckpt")


  else:
    restore_name = sys.argv[1]
    prediction_graph   = build_time_series_graph(GraphConfig())
    prediction_session = tf.Session(graph=prediction_graph)
# @TODO pass the proper dorput rate1
    with prediction_session as sess:
      saver = tf.train.Saver(max_to_keep=0)
      saver.restore(sess, restore_name)
      measure_performance(prediction_graph, sess, verification_X[0], verification_y[0])


if __name__ == "__main__":
  tf.app.run()


