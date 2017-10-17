#!/usr/bin/env python
from __future__ import division

import tensorflow as tf
import i3d
from inputs_new import *
from config import *
import argparse

# build the model
def inference(rgb_inputs, flow_inputs):
  with tf.variable_scope('RGB'):
    rgb_model = i3d.InceptionI3d(
      NUM_CLASSES, spatial_squeeze=True, final_endpoint='Logits')
    rgb_logits, _ = rgb_model(
      rgb_inputs, is_training=True, dropout_keep_prob=1.0)
  with tf.variable_scope('Flow'):
    flow_model = i3d.InceptionI3d(
        NUM_CLASSES, spatial_squeeze=True, final_endpoint='Logits')
    flow_logits, _ = flow_model(
        flow_inputs, is_training=True, dropout_keep_prob=1.0)
  return rgb_logits, flow_logits


def evaluate(input_file, ckpt_dir):
  with tf.Graph().as_default() as g:
    pipeline = InputPipeLine(input_file, num_epochs=1)
    rgbs, flows, labels = pipeline.get_batch(train=False)
    rgb_logits, flow_logits = inference(rgbs, flows)
    model_logits = rgb_logits + flow_logits
    batch_loss = tf.reduce_sum(
                   tf.nn.sparse_softmax_cross_entropy_with_logits(
                     labels=labels,logits=model_logits))
    top_k_op = tf.nn.in_top_k(model_logits, labels, 1)
    
    saver = tf.train.Saver()
    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())

      ckpt = tf.train.get_checkpoint_state(ckpt_dir)
      if ckpt and ckpt.model_checkpoint_path:
        print 'Restoring from:', ckpt.model_checkpoint_path
        saver.restore(sess, ckpt.model_checkpoint_path)
      else:
        print 'error restoring ckpt...'
        return
      
      coord = tf.train.Coordinator()
      threads = pipeline.start(sess, coord)
      
      true_count = 0
      accum_loss = 0
      try:
        i = 0 
        while not coord.should_stop():
          cnt, loss = sess.run([top_k_op, batch_loss])
          true_count += np.sum(cnt)
          accum_loss += loss
      except tf.errors.OutOfRangeError as e:
        coord.request_stop(e)

      coord.request_stop()
      coord.join(threads)

      acc = true_count / len(pipeline.videos)
      avg_loss = accum_loss / len(pipeline.videos)
      return acc, avg_loss

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('input_file')
  parser.add_argument('--ckpt_dir', required=True)
  args = parser.parse_args()  
  evaluate(args.input_file, args.ckpt_dir)
