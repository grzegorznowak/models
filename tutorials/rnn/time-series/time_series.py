import sys
import numpy as np
import tensorflow as tf

from random   import randint
from datetime import datetime
from time_series_data import get_shuffled_training_set


class GraphConfig(object):
  """Large config."""
  n_neurons     = 5
  batch_size    = 120
  n_inputs      = 6
  n_layers      = 2
  n_outputs     = 6
  learning_rate = 0.001
  keep_prob     = 0.5
  log_dir       = "{}/run-{}/".format('/tmp/time_series_logdir', datetime.utcnow().strftime("%Y%m%d%H%M%S"))


def build_time_series_graph(graph_config, is_training):

  graph = tf.Graph()
  with graph.as_default():
    create_lstm      = lambda: tf.contrib.rnn.LSTMCell(num_units=graph_config.n_neurons, use_peepholes=True)
    if is_training:
      # NOTE: DropoutWrapper does not support is_training flag, thus we do branching here !
      create_dropout = lambda cell: tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=graph_config.keep_prob)
    else:
      # just identity function if not training -> no extra layers added
      create_dropout = lambda x: x

    X              = tf.placeholder(tf.float32, [None, graph_config.batch_size, graph_config.n_inputs] , name="X")
    y              = tf.placeholder(tf.float32, [None, graph_config.batch_size, graph_config.n_outputs], name="y")
    cell_layers    = [create_lstm() for _ in range(graph_config.n_layers)]
    dropout_layers = list(map(create_dropout, cell_layers))

    multi_layer_cell     = tf.contrib.rnn.MultiRNNCell(dropout_layers)
    rnn_outputs, states  = tf.nn.dynamic_rnn(multi_layer_cell, X, dtype=tf.float32)

    stacked_rnn_outputs = tf.reshape(rnn_outputs, [-1, graph_config.n_neurons])
    stacked_outputs     = tf.layers.dense(stacked_rnn_outputs, graph_config.n_outputs)
    outputs             = tf.reshape(stacked_outputs,
                                     [-1, graph_config.batch_size, graph_config.n_outputs],
                                     name="outputs")

    loss             = tf.reduce_mean(tf.square(outputs - y), name="loss")
  #  optimizer       = tf.train.MomentumOptimizer(learning_rate=graph_config.learning_rate, momentum=0.9)
    optimizer        = tf.train.AdamOptimizer(learning_rate=graph_config.learning_rate)
    training_op      = optimizer.minimize(loss, name="training_op")

    # logging nodes
    mse_summary      = tf.summary.scalar("mse_summary", loss)


  return graph


def training_iteration(iteration, train_X, train_y, verify_x, verify_y, graph, session, saver, file_writer, batch_size):
  X           = graph.get_tensor_by_name("X:0")
  y           = graph.get_tensor_by_name("y:0")
  loss        = graph.get_tensor_by_name("loss:0")
  training_op = graph.get_operation_by_name("training_op")
  session.run(training_op, feed_dict={X: train_X, y: train_y})
  if iteration % 100 == 0:
    mse        = loss.eval(feed_dict={X: train_X, y: train_y})
    mse2       = loss.eval(feed_dict={X: train_X, y: train_y})
    verify_mse = loss.eval(feed_dict={X: verify_x, y: verify_y})
    merged  = tf.summary.merge_all()
    summary = session.run(merged, feed_dict={X: train_X, y: train_y})

    print(iteration, "\tMSE:" , mse)
    print(iteration, "\tMSE2:", mse2)
    print(iteration, "\tVerification MSE:", verify_mse)

    saver.save(session, "/tmp/time_series/model_" + str(iteration) + ".ckpt")
    file_writer.add_summary(summary, iteration)


def prediction(graph, X_batch, y_batch):
  X           = graph.get_tensor_by_name("X:0")
  outputs     = graph.get_tensor_by_name("outputs:0")
  print(X_batch)
  print(y_batch)
  print(outputs.eval(feed_dict={X: X_batch}))


def main(_):

  is_training      = (sys.argv[1] == "train")

  n_iterations     = 50000
  training_config  = GraphConfig()
  tain_X, train_y, verification_X, verification_y = get_shuffled_training_set(training_config.batch_size, 20)
  epoch_size       = len(tain_X)

  if is_training:
    training_graph   = build_time_series_graph(training_config, is_training=True)
    training_session = tf.Session(graph=training_graph)

    with training_session as sess:
      init        = tf.global_variables_initializer()
      saver       = tf.train.Saver()
      file_writer = tf.summary.FileWriter(training_config.log_dir, tf.get_default_graph())
      init.run()
      list(map(
        lambda iteration: training_iteration(iteration, tain_X[iteration % epoch_size], train_y[iteration % epoch_size], verification_X[0], verification_y[0], training_graph, sess, saver, file_writer, training_config.batch_size),
        range(n_iterations)))
      saver.save(sess, "/tmp/time_series/model_final.ckpt")

  else:
    restore_name = sys.argv[1]
    prediction_graph   = build_time_series_graph(GraphConfig(), is_training=False)
    prediction_session = tf.Session(graph=prediction_graph)

    with prediction_session as sess:
      saver = tf.train.Saver()
      saver.restore(sess, restore_name)
      prediction(prediction_graph, verification_X[0], verification_y[0])


if __name__ == "__main__":
  tf.app.run()
