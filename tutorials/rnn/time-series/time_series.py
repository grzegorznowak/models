import sys
import numpy as np
import tensorflow as tf
import time_series_data

from random   import randint
from datetime import datetime


class SmallGraphConfig(object):
  name          = "small"
  n_neurons     = 10
  batch_size    = 5
  n_inputs      = 6  # assumes datetime entry is present
  n_layers      = 3
  n_outputs     = 4
  initial_lr    = 0.001   #initial learning rate
  decay_lr      = 0.99
  keep_prob     = 0.9

class MediumGraphConfig(object):
  name           = "medium"
  rnn_neurons    = 200
  batch_size     = 60
  rnn_layers     = 2
  hidden_layers  = 3   # this is not automated still
  hidden_neurons = 10
  n_outputs      = 3
  n_inputs       = 4
  initial_lr     = 0.001   #initial learning rate
  decay_lr       = 0.9
  keep_prob      = 0.5     # droput only on RNN layer(s)

class MediumGraphConfig2(object):
  name           = "medium2"
  rnn_neurons    = 100
  batch_size     = 30
  rnn_layers     = 2
  n_outputs      = 3
  n_inputs       = 4
  initial_lr     = 0.001   #initial learning rate
  decay_lr       = 0.99
  keep_prob      = 0.95     # droput only on RNN layer(s)

class MediumGraphConfig3(object):
  name           = "medium3"
  rnn_neurons    = 100
  batch_size     = 60
  rnn_layers     = 2
  n_outputs      = 3
  n_inputs       = 4
  initial_lr     = 0.001   #initial learning rate
  decay_lr       = 0.99
  keep_prob      = 0.5     # droput only on RNN layer(s)


class MediumGraphConfig4(object):
  name           = "medium4"
  rnn_neurons    = 500
  batch_size     = 1
  rnn_layers     = 2
  n_outputs      = 4
  n_inputs       = 4
  initial_lr     = 0.001   #initial learning rate
  decay_lr       = 0.95
  keep_prob      = 0.99     # dropout only on RNN layer(s)

class MediumBigGraphConfig(object):
  name           = "medium-big"
  rnn_neurons    = 40
  batch_size     = 400
  rnn_layers     = 2
  hidden_layers  = 3   # this is not automated still
  hidden_neurons = 10
  n_outputs      = 3
  n_inputs       = 4
  initial_lr     = 0.0005   #initial learning rate
  decay_lr       = 0.9
  keep_prob      = 0.5     # droput only on RNN layer(s)


class GraphWrapper():
  def __init__(self, graph, init_op, initial_state_placeholder, multi_layer_cell, final_state_op):
     self.graph                     = graph
     self.init_op                   = init_op
     self.initial_state_placeholder = initial_state_placeholder
     self.multi_layer_cell          = multi_layer_cell
     self.final_state_op            = final_state_op

def build_rnn_time_series_graph(graph_config):

  create_rnn     = lambda:      tf.contrib.rnn.GRUCell(num_units=graph_config.rnn_neurons) #tf.nn.relu , use_peepholes=True
  create_dropout = lambda cell: tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=keep_prob)

  graph = tf.Graph()
  with graph.as_default():
    keep_prob      = tf.placeholder(tf.float32, None, name="keep_prob")
    he_init        = tf.contrib.layers.variance_scaling_initializer()

    X              = tf.placeholder(tf.float32, [None, graph_config.batch_size, graph_config.n_inputs] , name="X")
    y              = tf.placeholder(tf.float32, [None, graph_config.batch_size, graph_config.n_outputs], name="y")
    learning_rate  = tf.placeholder(tf.float32, None, name="learning_rate")

    cell_layers    = [create_rnn() for _ in range(graph_config.rnn_layers)]
    dropout_layers = list(map(create_dropout, cell_layers))

    multi_layer_cell          = tf.contrib.rnn.MultiRNNCell(dropout_layers, state_is_tuple=False)
    initial_state_placeholder = tf.placeholder(tf.float32, name="initial_state_placeholder")

    rnn_outputs, final_state   = tf.nn.dynamic_rnn(multi_layer_cell, X, dtype=tf.float32, initial_state=initial_state_placeholder)

    stacked_rnn_outputs = tf.reshape(rnn_outputs, [-1, graph_config.rnn_neurons], name="stacked_rnn_outputs")
    stacked_outputs     = tf.layers.dense(stacked_rnn_outputs, graph_config.n_outputs, name="stacked_outputs", kernel_initializer=he_init)
    outputs             = tf.reshape(stacked_outputs,
                                     [-1, graph_config.batch_size, graph_config.n_outputs],
                                     name="outputs")

    loss       = tf.reduce_mean(tf.square(outputs - y), name="loss")
    optimizer  = tf.train.AdamOptimizer(learning_rate=learning_rate)

  #  gvs        = optimizer.compute_gradients(loss)
  #  capped_gvs = [(tf.clip_by_value(grad, -20., 20.), var) for grad, var in gvs]
  #  training_op = optimizer.apply_gradients(capped_gvs, name="training_op")

    training_op    = optimizer.minimize(loss, name="training_op")



    mse_summary    = tf.summary.scalar("mse_summary", loss)
    tf.summary.histogram("stacked_outputs", stacked_outputs)
    tf.summary.histogram("weights_output", outputs)
    tf.summary.histogram("rnn_outputs", rnn_outputs)
    tf.summary.histogram("new_states", final_state)

    init = tf.global_variables_initializer()

  return GraphWrapper(graph, init, initial_state_placeholder, multi_layer_cell, final_state)

def training_iteration(previous_state, iteration, epoch, train_X, train_y, verify_X, verify_y, graph_wrapper, session, saver, save_dir, file_writer, training_config):
  graph         = graph_wrapper.graph
  initial_state_placeholder = graph_wrapper.initial_state_placeholder
  new_state_op = graph_wrapper.final_state_op
  X             = graph.get_tensor_by_name("X:0")
  y             = graph.get_tensor_by_name("y:0")
  keep_prob     = graph.get_tensor_by_name("keep_prob:0")
  loss          = graph.get_tensor_by_name("loss:0")
  outputs       = graph.get_tensor_by_name("outputs:0")
  learning_rate = graph.get_tensor_by_name("learning_rate:0")
  training_op   = graph.get_operation_by_name("training_op")

  current_learning_rate = training_config.initial_lr * (training_config.decay_lr**epoch)
  train, new_state = session.run([training_op, new_state_op] , feed_dict={X: train_X, y: train_y, keep_prob: training_config.keep_prob, learning_rate: current_learning_rate, initial_state_placeholder: previous_state})

  if iteration % 100 == 0:
    mse        = session.run(loss, feed_dict={X: train_X, y: train_y, keep_prob: 1, initial_state_placeholder: previous_state})
    verify_mse = session.run(loss, feed_dict={X: verify_X, y: verify_y, keep_prob: 1, initial_state_placeholder: previous_state})
    merged     = tf.summary.merge_all()
    summary    = session.run(merged, feed_dict={X: train_X, y: train_y, keep_prob: 1, initial_state_placeholder: previous_state})
    train_response  = session.run(outputs, feed_dict={X: train_X, keep_prob: 1, initial_state_placeholder: previous_state})
    verify_response = session.run(outputs, feed_dict={X: verify_X, keep_prob: 1, initial_state_placeholder: previous_state})
    file_writer.add_summary(summary, iteration)

    print("epoch: ", epoch, "iteration: ", iteration)
    print("train_y: \n"  , train_y)
    print("train_response: \n "  , train_response)
    print("verify_y:        "  , verify_y)
    print("verify_response: "  , verify_response)
    print("\tMSE:"             , mse)
    print("\tVerification MSE:", verify_mse)
    print("current LR: ", current_learning_rate)

  if iteration % 5000 == 0:  # save network rarily
    saver.save(session, save_dir + "model_" + str(iteration) + "_" + str(epoch) + ".ckpt")

  return new_state

def prediction(graph, session, X_batch, y_batch):
  X           = graph.get_tensor_by_name("X:0")
  y           = graph.get_tensor_by_name("y:0")
  outputs     = graph.get_tensor_by_name("outputs:0")
  loss        = graph.get_tensor_by_name("loss:0")
  mse         = loss.eval(feed_dict={X: X_batch, y: y_batch})
  print(X_batch)
  print(y_batch)
  print(session.run(outputs, feed_dict={X: X_batch}))
  print("MSE: ", mse)


def main(_):

  is_training  = (sys.argv[1] == "train")
  is_continue  = (sys.argv[1] == "continue")

  training_config  = MediumGraphConfig4()

  if is_training:
    graph_wrapper    = build_rnn_time_series_graph(training_config)
    training_session = tf.Session(graph=graph_wrapper.graph)
    init_op          = graph_wrapper.init_op
    epochs           = 500


    with training_session as sess:
      init_op.run()
      saver = tf.train.Saver(max_to_keep=0)
      data_batches_count = 2 # time_series_data.get_total_data_batches_count_in_folder()


      if is_continue:
        restore_name  = sys.argv[2]
        print(restore_name)
        saver.restore(sess, restore_name)

      print("data_batches_count in folder", data_batches_count)

      for epoch in range(epochs):
        log_dir     = "{}/run-{}-{}-{}/".format('/tmp/time_series_logdir', datetime.utcnow().strftime("%Y%m%d%H%M%S"), epoch, training_config.name)
        save_dir    = "{}/run-{}-{}-{}/".format('/tmp/time_series', datetime.utcnow().strftime("%Y%m%d%H%M%S"), epoch, training_config.name)
        epoch_iteration = 0
        for data_batch in range(data_batches_count):
          train_X, train_y, verification_X, verification_y = time_series_data.get_data_batch_from_folder(data_batch, training_config.batch_size,  1)
          data_batch_size  = len(train_X) - 1
          inital_state_placeholder = graph_wrapper.initial_state_placeholder


          previous_state = graph_wrapper.multi_layer_cell.zero_state(training_config.batch_size, tf.float32)
          previous_state_value = np.zeros((1,training_config.rnn_neurons * training_config.rnn_layers))

          print("Training ", epoch, "epoch, with train set len of: ", data_batch_size, " iterations, current data batch: ",data_batch, ' / ', data_batches_count)
          file_writer = tf.summary.FileWriter(log_dir, tf.get_default_graph())
          for data_iteration in range(data_batch_size):
            previous_state_value = training_iteration(previous_state_value, epoch_iteration, epoch, train_X[data_iteration % data_batch_size], train_y[data_iteration % data_batch_size], verification_X[0], verification_y[0], graph_wrapper, sess, saver, save_dir, file_writer, training_config)
            epoch_iteration += 1
        saver.save(sess, save_dir + "model_final_" + str(epoch) + ".ckpt")

  else:
    restore_name = sys.argv[1]
    prediction_graph   = build_time_series_graph(GraphConfig())
    prediction_session = tf.Session(graph=prediction_graph)
# @TODO pass the proper dorput rate1
    with prediction_session as sess:
      saver = tf.train.Saver(max_to_keep=0)
      saver.restore(sess, restore_name)
      prediction(prediction_graph, sess, verification_X[0], verification_y[0])


if __name__ == "__main__":
  tf.app.run()
