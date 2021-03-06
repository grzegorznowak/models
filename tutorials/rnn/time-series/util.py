# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Utilities for Grappler autoparallel optimizer."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.python.client  import device_lib
from tensorflow.core.framework import variable_pb2
from tensorflow.core.protobuf  import rewriter_config_pb2

FLAGS = tf.flags.FLAGS

flags = tf.flags
logging = tf.logging

flags.DEFINE_string(
    "model", "small",
    "A type of model. Possible options are: small, medium, large.")
flags.DEFINE_string("data_path", None,
                    "Where the training/test data is stored.")
flags.DEFINE_string("save_path", None,
                    "Model output directory.")
flags.DEFINE_bool("use_fp16", False,
                  "Train using 16-bit floats instead of 32bit floats")
flags.DEFINE_integer("num_gpus", 1,
                     "If larger than 1, Grappler AutoParallel optimizer "
                     "will create multiple training replicas with each GPU "
                     "running one replica.")
flags.DEFINE_string("rnn_mode", None,
                    "The low level implementation of lstm cell: one of "
                    "BASIC, and BLOCK, representing basic_lstm, "
                    "and lstm_block_cell classes.")
FLAGS = flags.FLAGS
BASIC = "basic"
BLOCK = "block"


def state_tuples(state, name, type):
  return import_state_tuples(
    state,
    with_prefix(name, type),
    num_replicas(name))


def initial_state_tuples(initial_state, name):
  return state_tuples(initial_state, name, "initial")


def final_state_tuples(final_state, name):
  return state_tuples(final_state, name, "final")


def num_replicas(set_name):
  return FLAGS.num_gpus if set_name == "Train" else 1


def get_lstm_cell(config):
  return tf.contrib.rnn.LSTMBlockCell(
    config.hidden_size, forget_bias=0.0)


def create_lstm_cell(is_training, config):
  return tf.contrib.rnn.MultiRNNCell(
    [get_maybe_dropout_cell(is_training, config) for _ in range(config.num_layers)], state_is_tuple=True)


def get_zero_state_for_the_cell(cell, config):
  return cell.zero_state(config.batch_size, data_type())


def export_ops(model, config):
  """Exports ops to collections."""
  ops = {with_prefix(model.name, "cost"): model.cost}
  if model.is_training:
    ops.update(lr=model.lr, new_lr=model.new_lr, lr_update=model.lr_update)
  for name, op in ops.items():
    tf.add_to_collection(name, op)
  export_state_tuples(create_zero_state_cell(model.is_training, config), model.initial_state_name)
  export_state_tuples(model.final_state                                , model.final_state_name)


def create_zero_state_cell(is_training, config):
  return get_zero_state_for_the_cell(create_lstm_cell(is_training, config), config)


def get_maybe_dropout_cell(is_training, config):
    # Slightly better results can be obtained with forget gate biases
    # initialized to 1 but the hyperparameters of the model would need to be
    # different than reported in the paper.
  cell = get_lstm_cell(config)
  if is_training and config.keep_prob < 1:
    return tf.contrib.rnn.DropoutWrapper(
      cell, output_keep_prob=config.keep_prob)
  else:
    return cell


def export_state_tuples(state, name):
  for state_tuple in state:
    tf.add_to_collection(name, state_tuple.c)
    tf.add_to_collection(name, state_tuple.h)


def import_state_tuples(state, state_name, model_name):
  restored = []
  for i in range(len(state) * num_replicas(model_name)):
    c = tf.get_collection_ref(state_name)[2 * i + 0]
    h = tf.get_collection_ref(state_name)[2 * i + 1]
    restored.append(tf.contrib.rnn.LSTMStateTuple(c, h))
  return tuple(restored)


def with_prefix(prefix, name):
  """Adds prefix to name."""
  return "/".join((prefix, name))


def with_autoparallel_prefix(replica_id, name):
  return with_prefix("AutoParallel-Replica-%d" % replica_id, name)


class UpdateCollection(object):
  """Update collection info in MetaGraphDef for AutoParallel optimizer."""

  def __init__(self, metagraph, model):
    self._metagraph = metagraph
    self.replicate_states(model.initial_state_name)
    self.replicate_states(model.final_state_name)
    self.update_snapshot_name("variables")
    self.update_snapshot_name("trainable_variables")

  def update_snapshot_name(self, var_coll_name):
    var_list = self._metagraph.collection_def[var_coll_name]
    for i, value in enumerate(var_list.bytes_list.value):
      var_def = variable_pb2.VariableDef()
      var_def.ParseFromString(value)
      # Somehow node Model/global_step/read doesn't have any fanout and seems to
      # be only used for snapshot; this is different from all other variables.
      if var_def.snapshot_name != "Model/global_step/read:0":
        var_def.snapshot_name = with_autoparallel_prefix(
            0, var_def.snapshot_name)
      value = var_def.SerializeToString()
      var_list.bytes_list.value[i] = value

  def replicate_states(self, state_coll_name):
    state_list = self._metagraph.collection_def[state_coll_name]
    num_states = len(state_list.node_list.value)
    for replica_id in range(1, FLAGS.num_gpus):
      for i in range(num_states):
        state_list.node_list.value.append(state_list.node_list.value[i])
    for replica_id in range(FLAGS.num_gpus):
      for i in range(num_states):
        index = replica_id * num_states + i
        state_list.node_list.value[index] = with_autoparallel_prefix(
            replica_id, state_list.node_list.value[index])


def auto_parallel(metagraph, model):
  from tensorflow.python.grappler import tf_optimizer
  rewriter_config = rewriter_config_pb2.RewriterConfig()
  rewriter_config.optimizers.append("autoparallel")
  rewriter_config.auto_parallel.enable = True
  rewriter_config.auto_parallel.num_replicas = FLAGS.num_gpus
  optimized_graph = tf_optimizer.OptimizeGraph(rewriter_config, metagraph)
  metagraph.graph_def.CopyFrom(optimized_graph)
  UpdateCollection(metagraph, model)


def list_gpus():
  return [x.name for x in device_lib.list_local_devices() if x.device_type == "GPU"]


def data_type():
  return tf.float16 if FLAGS.use_fp16 else tf.float32


def get_config():
  """Get model config."""
  config = None
  if FLAGS.model == "small":
    config = SmallConfig()
  elif FLAGS.model == "medium":
    config = MediumConfig()
  elif FLAGS.model == "large":
    config = LargeConfig()
  elif FLAGS.model == "test":
    config = TestConfig()
  else:
    raise ValueError("Invalid model: %s", FLAGS.model)
  if FLAGS.rnn_mode:
    config.rnn_mode = FLAGS.rnn_mode
  if FLAGS.num_gpus != 1 or tf.__version__ < "1.3.0" :
    config.rnn_mode = BASIC
  return config



class SmallConfig(object):
  """Small config."""
  init_scale = 0.1
  learning_rate = 1.0
  max_grad_norm = 5
  num_layers = 2
  num_steps = 20
  hidden_size = 200
  max_epoch = 4
  max_max_epoch = 13
  keep_prob = 1.0
  lr_decay = 0.5
  batch_size = 20
  vocab_size = 10000
  rnn_mode = BLOCK


class MediumConfig(object):
  """Medium config."""
  init_scale = 0.05
  learning_rate = 1.0
  max_grad_norm = 5
  num_layers = 2
  num_steps = 35
  hidden_size = 650
  max_epoch = 6
  max_max_epoch = 39
  keep_prob = 0.5
  lr_decay = 0.8
  batch_size = 20
  vocab_size = 10000
  rnn_mode = BLOCK


class LargeConfig(object):
  """Large config."""
  init_scale = 0.04
  learning_rate = 1.0
  max_grad_norm = 10
  num_layers = 2
  num_steps = 35
  hidden_size = 1500
  max_epoch = 14
  max_max_epoch = 55
  keep_prob = 0.35
  lr_decay = 1 / 1.15
  batch_size = 20
  vocab_size = 10000
  rnn_mode = BLOCK


class TestConfig(object):
  """Tiny config, for testing."""
  init_scale = 0.1
  learning_rate = 1.0
  max_grad_norm = 1
  num_layers = 1
  num_steps = 2
  hidden_size = 2
  max_epoch = 1
  max_max_epoch = 1
  keep_prob = 1.0
  lr_decay = 0.5
  batch_size = 20
  vocab_size = 10000
  rnn_mode = BLOCK
