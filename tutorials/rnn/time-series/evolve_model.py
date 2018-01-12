# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Example / benchmark for building a PTB LSTM model.

Trains the model described in:
(Zaremba, et. al.) Recurrent Neural Network Regularization
http://arxiv.org/abs/1409.2329

There are 3 supported model configurations:
===========================================
| config | epochs | train | valid  | test
===========================================
| small  | 13     | 37.99 | 121.39 | 115.91
| medium | 39     | 48.45 |  86.16 |  82.07
| large  | 55     | 37.87 |  82.62 |  78.29
The exact results may vary depending on the random initialization.

The hyperparameters used in the model:
- init_scale - the initial scale of the weights
- learning_rate - the initial value of the learning rate
- max_grad_norm - the maximum permissible norm of the gradient
- num_layers - the number of LSTM layers
- num_steps - the number of unrolled steps of LSTM
- hidden_size - the number of LSTM units
- max_epoch - the number of epochs trained with the initial learning rate
- max_max_epoch - the total number of epochs for training
- keep_prob - the probability of keeping weights in the dropout layer
- lr_decay - the decay of the learning rate for each epoch after "max_epoch"
- batch_size - the batch size
- rnn_mode - the low level implementation of lstm cell: one of
             BASIC, or BLOCK, representing basic_lstm, and
             lstm_block_cell classes.

The data required for this example is in the data/ dir of the
PTB dataset from Tomas Mikolov's webpage:

$ wget http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz
$ tar xvf simple-examples.tgz

To run:

$ python evolve_model.py --data_path=simple-examples/data/

"""
# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

import time

import numpy as np
import tensorflow as tf

import reader
import util

class PTBInput(object):
  """The input data."""

  def __init__(self, config, data, name=None):
    self.batch_size = batch_size = config.batch_size
    self.num_steps  = num_steps = config.num_steps
    self.epoch_size = ((len(data) // batch_size) - 1) // num_steps
    self.input_data, self.targets = reader.ptb_producer(
        data, batch_size, num_steps, name=name)


class PTBModel(object):
  """The PTB model."""

  def __init__(self, name, is_training, config, input_):
    self.is_training = is_training
    self.__input      = input_
    self._cell        = None
    self.name         = name
    self.batch_size   = input_.batch_size
    self.num_steps    = input_.num_steps
    size              = config.hidden_size
    vocab_size        = config.vocab_size

    self.initial_state_name = util.with_prefix(self.name, "initial")
    self.final_state_name   = util.with_prefix(self.name, "final")

    with tf.device("/cpu:0"):
      embedding = tf.get_variable(
          "embedding", [vocab_size, size], dtype=util.data_type())
      inputs = tf.nn.embedding_lookup(embedding, input_.input_data)

    if is_training and config.keep_prob < 1:
      inputs = tf.nn.dropout(inputs, config.keep_prob)

    output, state = self._build_rnn_graph_lstm(inputs, config, is_training)

    softmax_w = tf.get_variable("softmax_w", [size, vocab_size], dtype=util.data_type())
    softmax_b = tf.get_variable("softmax_b", [vocab_size]      , dtype=util.data_type())
    logits    = tf.nn.xw_plus_b(output, softmax_w, softmax_b)
    logits_3d = tf.reshape(logits, [self.batch_size, self.num_steps, vocab_size])

    # Use the contrib sequence loss and average over the batches
    loss = tf.contrib.seq2seq.sequence_loss(
        logits_3d,
        input_.targets,
        tf.ones([self.batch_size, self.num_steps], dtype=util.data_type()),
        average_across_timesteps=False,
        average_across_batch=True)

    # Update the cost
    self.cost        = tf.reduce_sum(loss)
    self.final_state = state

    if not is_training:
      return

    self.lr      = tf.Variable(0.0, trainable=False)
    tvars          = tf.trainable_variables()
    grads, _       = tf.clip_by_global_norm(tf.gradients(self.cost, tvars),
                                            config.max_grad_norm)
    tf.train.GradientDescentOptimizer(self.lr).apply_gradients(
      zip(grads, tvars),
      global_step=tf.train.get_or_create_global_step())

    self.new_lr    = tf.placeholder(tf.float32, shape=[], name="new_learning_rate")
    self.lr_update = tf.assign(self.lr, self.new_lr)

  def _build_rnn_graph_lstm(self, inputs, config, is_training):
    """Build the inference graph using canonical LSTM cells."""
    cell  = util.create_lstm_cell(is_training, config)
    state = util.get_zero_state_for_the_cell(cell, config)

    self.initial_state = state
    # Simplified version of tensorflow_models/tutorials/rnn/rnn.py's rnn().
    # This builds an unrolled LSTM for tutorial purposes only.
    # In general, use the rnn() or state_saving_rnn() from rnn.py.
    #
    # The alternative version of the code below is:
    #
    # inputs = tf.unstack(inputs, num=num_steps, axis=1)
    # outputs, state = tf.contrib.rnn.static_rnn(cell, inputs,
    #                            initial_state=self._initial_state)
    outputs = []
    with tf.variable_scope("RNN"):
      for time_step in range(self.num_steps):
        if time_step > 0:
          tf.get_variable_scope().reuse_variables()
        (cell_output, state) = cell(inputs[:, time_step, :], state)
        outputs.append(cell_output)
    output = tf.reshape(tf.concat(outputs, 1), [-1, config.hidden_size])
    return output, state

  def assign_lr(self, session, lr_value):
    session.run(self.lr_update, feed_dict={self.new_lr: lr_value})

  @property
  def input(self):
    return self.__input

  def import_ops(self):
    """Imports ops from collections."""
    if self.is_training:
      self.__train_op  = tf.get_collection_ref("train_op")[0]
      self.lr          = tf.get_collection_ref("lr")[0]
      self.new_lr      = tf.get_collection_ref("new_lr")[0]
      self.lr_update   = tf.get_collection_ref("lr_update")[0]

    self.cost          = tf.get_collection_ref(util.with_prefix(self.name, "cost"))[0]
    self.initial_state = util.import_state_tuples(
      self.initial_state, self.initial_state_name, self.name)
    self.final_state = util.import_state_tuples(
      self.final_state, self.final_state_name, self.name)

  @property
  def train_op(self):
    return self.__train_op


def run_epoch(session, model, eval_op=None, verbose=False):
  """Runs the model on the given data."""
  start_time = time.time()
  costs      = 0.0
  iters      = 0
  state      = session.run(model.initial_state)

  fetches = {
      "cost"       : model.cost,
      "final_state": util.final_state_tuples(model.final_state, model.name),
  }

  if eval_op is not None:
    fetches["eval_op"] = eval_op

  for step in range(model.input.epoch_size):
    feed_dict = {}
    for i, (c, h) in enumerate(util.initial_state_tuples(model.initial_state, model.name)):
      feed_dict[c] = state[i].c
      feed_dict[h] = state[i].h

    vals  = session.run(fetches, feed_dict)
    cost  = vals["cost"]
    state = vals["final_state"]

    costs += cost
    iters += model.input.num_steps

    if verbose and step % (model.input.epoch_size // 10) == 10:
      print("%.3f perplexity: %.3f speed: %.0f wps" %
            (step * 1.0 / model.input.epoch_size, np.exp(costs / iters),
             iters * model.input.batch_size * max(1, util.FLAGS.num_gpus) /
             (time.time() - start_time)))

  return np.exp(costs / iters)


def main(_):
  if not util.FLAGS.data_path:
    raise ValueError("Must set --data_path to PTB data directory")
  gpus = util.list_gpus()
  if util.FLAGS.num_gpus > len(gpus):
    raise ValueError(
        "Your machine has only %d gpus "
        "which is less than the requested --num_gpus=%d."
        % (len(gpus), util.FLAGS.num_gpus))

  train_data, valid_data, test_data, _ = reader.ptb_raw_data(util.FLAGS.data_path)

  config                 = util.get_config()
  eval_config            = util.get_config()
  eval_config.batch_size = 1
  eval_config.num_steps  = 1

  with tf.Graph().as_default():
    initializer = tf.random_uniform_initializer(-config.init_scale,
                                                config.init_scale)

    with tf.name_scope("Train"):
      train_input = PTBInput(config=config, data=train_data, name="TrainInput")
      with tf.variable_scope("Model", reuse=None, initializer=initializer):
        mtrain = PTBModel(name="Train", is_training=True, config=config, input_=train_input)
      tf.summary.scalar("Training Loss", mtrain.cost)
      tf.summary.scalar("Learning Rate", mtrain.lr)

    with tf.name_scope("Valid"):
      valid_input = PTBInput(config=config, data=valid_data, name="ValidInput")
      with tf.variable_scope("Model", reuse=True, initializer=initializer):
        mvalid = PTBModel(name="Valid", is_training=False, config=config, input_=valid_input)
      tf.summary.scalar("Validation Loss", mvalid.cost)

    with tf.name_scope("Test"):
      test_input = PTBInput(
          config=eval_config, data=test_data, name="TestInput")
      with tf.variable_scope("Model", reuse=True, initializer=initializer):
        mtest = PTBModel(name="Test", is_training=False, config=eval_config,
                         input_=test_input)

    models = [mtrain, mvalid, mtest]
    [(util.export_ops(model, config)) for model in models]
    metagraph = tf.train.export_meta_graph()

  with tf.Graph().as_default():

    tf.train.import_meta_graph(metagraph)
    [(model.import_ops()) for model in models]

    sv           = tf.train.Supervisor(logdir=util.FLAGS.save_path)
    config_proto = tf.ConfigProto(allow_soft_placement=False)
    with sv.managed_session(config=config_proto) as session:
      for i in range(config.max_max_epoch):
        lr_decay = config.lr_decay ** max(i + 1 - config.max_epoch, 0.0)
        mtrain.assign_lr(session, config.learning_rate * lr_decay)

        print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(mtrain.lr)))
        train_perplexity = run_epoch(session, mtrain, eval_op=mtrain.train_op,
                                     verbose=True)
        print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))
        valid_perplexity = run_epoch(session, mvalid)
        print("Epoch: %d Valid Perplexity: %.3f" % (i + 1, valid_perplexity))

      test_perplexity = run_epoch(session, mtest)
      print("Test Perplexity: %.3f" % test_perplexity)

      if util.FLAGS.save_path:
        print("Saving model to %s." % util.FLAGS.save_path)
        sv.saver.save(session, util.FLAGS.save_path, global_step=sv.global_step)


if __name__ == "__main__":
  tf.app.run()
