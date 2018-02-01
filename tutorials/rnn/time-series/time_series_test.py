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

"""Tests for models.tutorials.rnn.times-series.reader."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path

import tensorflow as tf
from time_series_data import get_random_data_batch_from_folder


class TimesSeriesReaderTest(tf.test.TestCase):

  def setUp(self):

    self.RawData      = [['0.8890866', '0.8727374', '0.8890866', '0.8738766', '-1.000', '0.9166667'],
                        ['0.4000000', '0.2408240', '0.4000000', '0.2408240', '-1.000', '0.9180556'],
                        ['0.1204120', '0.3380392', '0.0000000', '0.3380392', '-1.000', '0.9208333']]
    self._string_data = '\n'.join(map(','.join, self.RawData))


  def testCsvReading(self):

    for _ in range(10):
      (X, y), random_index = get_random_data_batch_from_folder(4)

      print(X[-1])
      print(y[-1])
      print(random_index)



if __name__ == "__main__":
  tf.test.main()
