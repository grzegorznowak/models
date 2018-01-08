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
import csv
import reader


class TimesSeriesReaderTest(tf.test.TestCase):

  def setUp(self):

    self.RawData      = [['0.8890866', '0.8727374', '0.8890866', '0.8738766', '-1.000', '0.9166667'],
                        ['0.4000000', '0.2408240', '0.4000000', '0.2408240', '-1.000', '0.9180556'],
                        ['0.1204120', '0.3380392', '0.0000000', '0.3380392', '-1.000', '0.9208333']]
    self._string_data = '\n'.join(map(','.join, self.RawData))


  def testCsvReading(self):
    tmpdir = tf.test.get_temp_dir()

    filename = os.path.join(tmpdir, "timeseries_test.txt")
    with tf.gfile.GFile(filename, "w") as fh:
      fh.write(self._string_data)

    csv_data = []
    with open(filename, newline='') as f:
      reader = csv.reader(f)
      for row in reader:
        csv_data.append(row)

    self.assertAllEqual(self.RawData, csv_data)

  def testParsingDataIntoVectors(self):
    csv_data =[]

    for row in csv.reader('0.8890866,0.8727374,0.8890866,0.8738766,-1.000,0.9166667\n'
                           +'0.4000000,0.2408240,0.4000000,0.2408240,-1.000,0.9180556\n'
                           +'0.1204120,0.3380392,0.0000000,0.3380392,-1.000,0.9208333',  delimiter=',', quoting=csv.QUOTE_NONE):

      csv_data.append(row)
      print(row)
    # raw_data = [4, 3, 2, 1, 0, 5, 6, 1, 1, 1, 1, 0, 3, 4, 1]
    batch_size = 3
    num_steps = 2
    x, y = reader.ptb_producer(raw_data, batch_size, num_steps)
    with self.test_session() as session:
      coord = tf.train.Coordinator()
      tf.train.start_queue_runners(session, coord=coord)
      try:
        xval, yval = session.run([x, y])
        self.assertAllEqual(xval, [[4, 3], [5, 6], [1, 0]])
        self.assertAllEqual(yval, [[3, 2], [6, 1], [0, 3]])
        xval, yval = session.run([x, y])
        self.assertAllEqual(xval, [[2, 1], [1, 1], [3, 4]])
        self.assertAllEqual(yval, [[1, 0], [1, 1], [4, 1]])
      finally:
        coord.request_stop()
        coord.join()


if __name__ == "__main__":
  tf.test.main()
