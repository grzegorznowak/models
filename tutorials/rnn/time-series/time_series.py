import gc
import sys
import numpy as np
import tensorflow as tf
import time_series_data

from random   import randint
from datetime import datetime
from pympler import tracker
from pympler import summary
from pympler import muppy
from functools import lru_cache

tr = tracker.SummaryTracker()

class LaptopCPUConfig(object):
  name           = "LaptopCPU"
  rnn_neurons    = 1000
  batch_size     = 1
  rnn_layers     = 2
  n_outputs      = 4
  n_inputs       = 4
  initial_lr     = 0.001   #initial learning rate
  decay_lr       = 0.99
  keep_prob      = 0.5     # dropout only on RNN layer(s)
  def create_rnn(self):
    return tf.contrib.rnn.RNNCell(num_units=self.rnn_neurons) # try using faster cells


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

  def create_rnn(self):
    return tf.contrib.rnn.GRUCell(num_units=self.rnn_neurons) #tf.nn.relu , use_peepholes=True

class DesktopCPUConfig2(object):
  name           = "DesktopCPU2"
  rnn_neurons    = 500
  batch_size     = 10
  rnn_layers     = 3
  n_inputs       = 6
  n_outputs      = 1
  initial_lr     = 0.001   #initial learning rate
  decay_lr       = 0.95
  keep_prob      = 0.5     # dropout only on RNN layer(s)

  def create_rnn(self):
    return tf.contrib.rnn.GRUCell(num_units=self.rnn_neurons) # try using faster cells


class GraphWrapper():
  def __init__(self, graph, init_op, initial_state_placeholder, multi_layer_cell,
               final_state_op, train_day, X, y, epoch, loss, outputs, learning_rate,
               training_op, keep_prob, summary_op, rnn_outputs, final_state, stacked_rnn_outputs,
               stacked_outputs, is_training, saver):
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
     self.outputs_placeholder       = outputs
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


def build_rnn_time_series_graph(graph_config):

  create_dropout = lambda cell: tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=keep_prob)

  graph = tf.Graph()
  with graph.as_default():
    keep_prob      = tf.placeholder(tf.float32, None, name="keep_prob")
    he_init        = tf.contrib.layers.variance_scaling_initializer()

    X              = tf.placeholder(tf.float32, [None, graph_config.batch_size, graph_config.n_inputs] , name="X")
    y              = tf.placeholder(tf.float32, [None, graph_config.batch_size, graph_config.n_outputs], name="y")
    learning_rate  = tf.placeholder(tf.float32, None, name="learning_rate")
    epoch          = tf.placeholder(tf.int16  , name="epoch")
    train_day      = tf.placeholder(tf.int16  , name="train_day")
    is_training    = tf.placeholder_with_default(True  , shape=(), name="is_training")
    initial_state_placeholder  = tf.placeholder(tf.float32, [1, graph_config.rnn_neurons * graph_config.rnn_layers], name="initial_state_placeholder")

    cell_layers    = [graph_config.create_rnn() for _ in range(graph_config.rnn_layers)]
    dropout_layers = list(map(create_dropout, cell_layers))

    hidden_input_0             = tf.layers.dense(X, graph_config.rnn_neurons)  # squash inputs for further processing
    hidden_normalization_0     = tf.contrib.layers.layer_norm(hidden_input_0)
    hidden_normalization_0_act = tf.nn.elu(hidden_normalization_0)
    multi_layer_cell           = tf.contrib.rnn.MultiRNNCell(dropout_layers, state_is_tuple=False)

    input_state_preproces      = tf.layers.dense(initial_state_placeholder, graph_config.rnn_neurons * graph_config.rnn_layers, activation=tf.sigmoid)  # sigmoid to have it output proper values for RNN's state values
    rnn_outputs, final_state   = tf.nn.dynamic_rnn(multi_layer_cell, hidden_normalization_0_act, dtype=tf.float32, initial_state=input_state_preproces)


    stacked_rnn_outputs = tf.reshape(rnn_outputs, [-1, graph_config.rnn_neurons], name="stacked_rnn_outputs")

 #   output_normalization_0     = tf.contrib.layers.layer_norm(stacked_rnn_outputs)
 #   output_normalization_0_act = tf.nn.elu(output_normalization_0)
  #  hidden_outputs_0    = tf.layers.dense(stacked_rnn_outputs, graph_config.rnn_neurons)
   # hidden_outputs_1    = tf.layers.dense(hidden_outputs_0, graph_config.rnn_neurons // 2)
   # hidden_outputs_2    = tf.layers.dense(hidden_outputs_1, graph_config.rnn_neurons // 4)   # unsquash for output values
    stacked_outputs     = tf.layers.dense(stacked_rnn_outputs, graph_config.n_outputs, name="stacked_outputs")
    outputs             = tf.reshape(stacked_outputs,
                                     [-1, graph_config.batch_size, graph_config.n_outputs],
                                     name="outputs")
    outputs_int  = tf.cast(outputs, tf.int32)
    loss       = tf.losses.mean_squared_error(y , outputs)
    optimizer  = tf.train.AdamOptimizer(learning_rate=learning_rate)
    #optimizer  = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)

  #  gvs        = optimizer.compute_gradients(loss)
  #  capped_gvs = [(tf.clip_by_value(grad, -20., 20.), var) for grad, var in gvs]
  #  training_op = optimizer.apply_gradients(capped_gvs, name="training_op")

    training_op    = optimizer.minimize(loss, name="training_op")


    tf.summary.scalar("learning_rate", learning_rate)
    tf.summary.scalar("epoch"        , epoch)
    tf.summary.histogram("train_day"    , train_day)
    tf.summary.scalar("mse_summary"  , loss)

    tf.summary.histogram("weights_output", outputs)
    tf.summary.histogram("new_states", final_state)

    init = tf.global_variables_initializer()

    summary_op = tf.summary.merge_all()

    saver = tf.train.Saver(max_to_keep=0)

  return GraphWrapper(graph, init, initial_state_placeholder, multi_layer_cell, final_state,
                      train_day, X, y, epoch, loss, outputs_int, learning_rate, training_op,
                      keep_prob, summary_op, rnn_outputs, final_state, stacked_rnn_outputs,
                      stacked_outputs, is_training, saver)



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

    if batch_number % 10 == 0: # print out details only every 10
      print("Verification batch number: ", batch_number,
          "\tvY vs vResp:\t",
            np.transpose(y[batch_number][-1]), "\t<->",  np.transpose(train_response[-1]),
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
  # rnn_outputs_op = graph_wrapper.rnn_outputs
  # final_state_op = graph_wrapper.final_state
  # stacked_rnn_outputs_op = graph_wrapper.stacked_rnn_outputs
  # stacked_outputs_op = graph_wrapper.stacked_outputs

  train, new_state = session.run([training_op, new_state_op],
                                      feed_dict={X: train_X, y: train_y, keep_prob: training_config.keep_prob,
                                                 learning_rate: current_learning_rate,
                                                 initial_state_placeholder: previous_state})

  if iteration % 1000 == 0:
    summary = session.run(graph_wrapper.summary_op,
                          feed_dict={
                             X: train_X, y: train_y, keep_prob: 1,
                             initial_state_placeholder: previous_state, learning_rate: current_learning_rate,
                             epoch_holder: epoch,
                             train_day_placeholder: random_index,
                             is_training_placeholder: False})

    file_writer.add_summary(summary, iteration)

  if iteration % 2000 == 0:
    mse, train_response             = session.run([loss, outputs],
                                                  feed_dict={X: train_X, y: train_y, keep_prob: 1,
                                                             initial_state_placeholder: previous_state,
                                                             is_training_placeholder: False})
   # verify_mse     = session.run(loss, feed_dict={X: verify_X, y: verify_y, keep_prob: 1, initial_state_placeholder: previous_state})
    #train_response  = session.run(outputs, feed_dict={X: train_X, keep_prob: 1, initial_state_placeholder: previous_state})

    print("epoch: ", epoch, ", (ticks) iteration: ", iteration)
    print("last train_y vs output: \n"  , np.transpose(train_y[-1]), "\t -> \t", np.transpose(train_response[-1]), "\tMSE:"  , mse)
    print("current LR: ", current_learning_rate)

  if iteration % 150000 == 0:  # save network rarily
    saver.save(session, save_dir + "model_" + str(iteration) + "_" + str(epoch) + ".ckpt")

  return new_state

def main(_):

  is_training, is_continue, restore_name, start_day_input, end_day_input = time_series_data.parse_cmdline(sys.argv)

  if(len(sys.argv) < 1):
    print("wrong usage")
    os.exit(1)

  training_config  = DesktopCPUConfig2()

  # @TODO: need to redo those CMD params logic when they grow in number. Just stick to the bruteforce IF power
  if is_training or is_continue:

    epochs           = 500
    graph_wrapper    = build_rnn_time_series_graph(training_config)
    init_op          = graph_wrapper.init_op
    saver            = graph_wrapper.saver
    training_session = tf.Session(graph=graph_wrapper.graph)


    with training_session as sess:

      data_batches_count = time_series_data.get_total_data_batches_count_in_train_folder()

      # it doesn't mean that much anymore, but is a good heuristic to skip to another epoch after
      # data_batches_count worth of samples has passed by
      end_day            = data_batches_count

      if is_continue:
        print(restore_name)
        saver.restore(sess, restore_name)

      else:
        init_op.run()

      print("data_batches_count in folder", data_batches_count)

    #  @lru_cache(maxsize=2)
      def zero_state():
        #return np.zeros((1,training_config.rnn_neurons * training_config.rnn_layers))
        return np.random.rand(1,training_config.rnn_neurons * training_config.rnn_layers)

      start_day = int(start_day_input)
      if int(end_day_input):
        end_day =  int(end_day_input)
      for epoch in range(epochs):
        log_dir     = "{}/run-{}-{}-{}/".format('/tmp/time_series_logdir', datetime.utcnow().strftime("%Y%m%d%H%M%S"), epoch, training_config.name)
        save_dir    = "{}/run-{}-{}-{}/".format('/tmp/time_series', datetime.utcnow().strftime("%Y%m%d%H%M%S"), epoch, training_config.name)
        file_writer = tf.summary.FileWriter(log_dir, tf.get_default_graph())
        epoch_iteration = 0
        current_learning_rate = training_config.initial_lr * (training_config.decay_lr**epoch)
        (verification_X, verification_y), verification_day_index = time_series_data.get_random_data_batch_from_verification_folder(training_config.batch_size)

        for data_batch in range(start_day, end_day):
          train_X, train_y = time_series_data.get_train_data_batch_from_folder(data_batch, training_config.batch_size)
          data_batch_size      = len(train_X) - 1
          previous_state_value = zero_state()

          print("Training ", epoch, "epoch, \tlearning rate:", current_learning_rate,"\ttrain set len of: ", data_batch_size,
                " iterations, \ncurrent data batch: ",data_batch, ' / ', data_batches_count, ' simulating day no. ', data_batch)


          for data_iteration in range(data_batch_size):
            previous_state_value = training_iteration(previous_state_value, current_learning_rate, epoch_iteration, epoch, data_batch, train_X[data_iteration % data_batch_size], train_y[data_iteration % data_batch_size], graph_wrapper, sess, saver, save_dir, file_writer, training_config)
            epoch_iteration += training_config.batch_size

          if data_batch % 5 == 0:
            # print out verfication stats every a few days worth of training
            measure_performance(zero_state(), verification_X, verification_y, graph_wrapper, sess, verification_day_index)

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


