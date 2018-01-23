import sys
import numpy as np
import tensorflow as tf

from random   import randint
from datetime import datetime
from time_series_data import get_shuffled_training_set


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
  lstm_neurons   = 2
  batch_size     = 10
  lstm_layers    = 1
  hidden_layers  = 3   # this is not automated still
  hidden_neurons = 10
  n_outputs      = 3
  n_inputs       = 4
  initial_lr     = 0.001   #initial learning rate
  decay_lr       = 0.9
  keep_prob      = 0.7     # droput only on LTSM layer


def build_time_series_graph(graph_config):

  graph = tf.Graph()
  with graph.as_default():

    keep_prob      = tf.placeholder(tf.float32, None, name="keep_prob")
    he_init        = tf.contrib.layers.variance_scaling_initializer()

    #he_init = tf.contrib.layers.variance_scaling_initializer()
    create_lstm   = lambda:  tf.contrib.rnn.LSTMCell(num_units=graph_config.lstm_neurons, use_peepholes=True, activation=None) #tf.nn.relu
    #create_lstm    = lambda:  tf.nn.rnn_cell.LSTMCell(
    #                              num_units=graph_config.n_neurons, activation=tf.nn.relu)

    create_dropout = lambda cell: tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=keep_prob)

    X              = tf.placeholder(tf.float32, [None, graph_config.batch_size, graph_config.n_inputs] , name="X")
    y              = tf.placeholder(tf.float32, [None, 1, graph_config.n_outputs], name="y")
    learning_rate  = tf.placeholder(tf.float32, None, name="learning_rate")

    hidden1        = tf.layers.dense(inputs=X      , units=graph_config.hidden_neurons, activation=None, kernel_initializer=he_init)
    hidden2        = tf.layers.dense(inputs=hidden1, units=graph_config.hidden_neurons, activation=None, kernel_initializer=he_init)
    hidden3        = tf.layers.dense(inputs=hidden2, units=graph_config.hidden_neurons, activation=None, kernel_initializer=he_init, name="hidden3")

    cell_layers    = [create_lstm() for _ in range(graph_config.lstm_layers)]
    dropout_layers = list(map(create_dropout, cell_layers))

    multi_layer_cell     = tf.contrib.rnn.MultiRNNCell(dropout_layers)
    rnn_outputs, states  = tf.nn.dynamic_rnn(multi_layer_cell, hidden3, dtype=tf.float32)
  #  hidden_out1          = tf.layers.dense(inputs=rnn_outputs, units=graph_config.n_neurons, activation=None, kernel_initializer=he_init)
  #  dense2                = tf.layers.dense(inputs=dense1, units=graph_config.n_neurons, activation=None, kernel_initializer=he_init)
 #   dense3                = tf.layers.dense(inputs=dense2, units=graph_config.n_neurons, activation=None, kernel_initializer=he_init)

    stacked_rnn_outputs = tf.reshape(rnn_outputs, [-1, graph_config.lstm_neurons], name="stacked_rnn_outputs")
    stacked_outputs     = tf.layers.dense(stacked_rnn_outputs, graph_config.n_outputs, name="stacked_outputs")
    outputs             = tf.reshape(stacked_outputs,
                                     [-1, graph_config.batch_size, graph_config.n_outputs],
                                     name="outputs")
    last_output =tf.squeeze(tf.transpose(outputs, [0, 2, 1])[:,:,4], name="last_output") # get last row - Shape of [batch_size, cell_units]

    loss             = tf.reduce_mean(tf.square(last_output - y), name="loss")
  #  optimizer       = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
    optimizer        = tf.train.AdamOptimizer(learning_rate=learning_rate)
  #  optimizer        = tf.train.RMSPropOptimizer(learning_rate)
    training_op      = optimizer.minimize(loss, name="training_op")

    # logging nodes
    mse_summary      = tf.summary.scalar("mse_summary", loss)
    tf.summary.histogram("weights_hidden1", hidden1)
    tf.summary.histogram("weights_hidden2", hidden2)
    tf.summary.histogram("weights_hidden3", hidden3)
 #  tf.summary.histogram("hidden_out1", hidden_out1)
    tf.summary.histogram("stacked_outputs", stacked_outputs)



  return graph


def training_iteration(iteration, epoch, train_X, train_y, verify_X, verify_y, graph, session, saver, save_dir, file_writer, training_config):
  X             = graph.get_tensor_by_name("X:0")
  y             = graph.get_tensor_by_name("y:0")
  keep_prob     = graph.get_tensor_by_name("keep_prob:0")
  loss          = graph.get_tensor_by_name("loss:0")
  output        = graph.get_tensor_by_name("last_output:0")
  learning_rate = graph.get_tensor_by_name("learning_rate:0")
  training_op   = graph.get_operation_by_name("training_op")

  current_learning_rate = training_config.initial_lr * (training_config.initial_lr**epoch)

  session.run(training_op, feed_dict={X: train_X, y: train_y, keep_prob: training_config.keep_prob, learning_rate: current_learning_rate})
  if iteration % 1000 == 0:
    mse        = session.run(loss, feed_dict={X: train_X, y: train_y, keep_prob: 1})
    verify_mse = session.run(loss, feed_dict={X: verify_X, y: verify_y, keep_prob: 1})
    merged     = tf.summary.merge_all()
    summary    = session.run(merged, feed_dict={X: train_X, y: train_y, keep_prob: 1})
    train_response  = session.run(output, feed_dict={X: train_X, keep_prob: 1})
    verify_response = session.run(output, feed_dict={X: verify_X, keep_prob: 1})
    train_X = session.run(X, feed_dict={X: train_X, keep_prob: 1})

    print("epoch: ", epoch, "iteration: ", iteration)
    print("train_y:         "  , train_y)
    print("train_response:  "  , train_response)
    print("verify_y:        "  , verify_y)
    print("verify_response: "  , verify_response)
    print(train_X)
    print("\tMSE:"             , mse)
    print("\tVerification MSE:", verify_mse)

    saver.save(session, save_dir + "model_" + str(iteration) + "_" + str(epoch) + ".ckpt")
    file_writer.add_summary(summary, iteration)


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

  training_config  = MediumGraphConfig()
  train_X, train_y, verification_X, verification_y = get_shuffled_training_set(training_config.batch_size, 5)
  epoch_size       = len(train_X) - 1
  epochs           = 500

  if is_training:
    training_graph   = build_time_series_graph(training_config)
    training_session = tf.Session(graph=training_graph)
    n_iterations     = epoch_size * epochs
    print("Training with train set len of: ", n_iterations, " iterations")


    with training_session as sess:
      init        = tf.global_variables_initializer()
      saver       = tf.train.Saver(max_to_keep=0)

      if is_continue:
        restore_name  = sys.argv[2]
        print(restore_name)
        saver.restore(sess, restore_name)
      else:
        init.run()

      for epoch in range(epochs):
        log_dir     = "{}/run-{}-{}-{}/".format('/tmp/time_series_logdir', datetime.utcnow().strftime("%Y%m%d%H%M%S"), epoch, training_config.name)
        save_dir    = "{}/run-{}-{}-{}/".format('/tmp/time_series', datetime.utcnow().strftime("%Y%m%d%H%M%S"), epoch, training_config.name)
        file_writer = tf.summary.FileWriter(log_dir, tf.get_default_graph())
        list(map(
          lambda iteration: training_iteration(iteration, epoch, train_X[iteration], train_y[iteration], verification_X[0], verification_y[0], training_graph, sess, saver, save_dir, file_writer, training_config),
          range(epoch_size)))
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
