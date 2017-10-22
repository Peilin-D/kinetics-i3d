#!/usr/bin/env python
from __future__ import division

import tensorflow as tf
import i3d
from inputs import *
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

def evaluate(input_file, ckpt_dir, top_k=None):
  """ 
  input_file: a text file containing absolute paths for test videos, one for each line
  ckpt_dir: model checkpoint directory
  top: (optional) if specified, will output top k predicted class with prob for each test video to out_prob.txt
  """
  with tf.Graph().as_default() as g:
    pipeline = InputPipeLine(input_file, num_epochs=1)
    rgbs, flows, labels = pipeline.get_batch(train=False)
    rgb_logits, flow_logits = inference(rgbs, flows)
    model_logits = rgb_logits + flow_logits
    top_k_op = tf.nn.in_top_k(model_logits, labels, 1)

    if top_k:
      prob_op = tf.nn.softmax(model_logits)
      cls_dict = {}
      with open('ucfTrainTestlist/classInd.txt', 'r') as f:
        for line in f.readlines():
          line = line.strip()
          ind, cls_name = line.split(' ')
          cls_dict[int(ind) - 1] = cls_name.lower()

    saver = tf.train.Saver()
    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())

      ckpt = tf.train.get_checkpoint_state(ckpt_dir)
      if ckpt and ckpt.model_checkpoint_path:
        print 'Restoring from:', ckpt.model_checkpoint_path
        saver.restore(sess, ckpt.model_checkpoint_path)
        print 'Restore Complete.'
      else:
        print 'error restoring ckpt...'
        return

      coord = tf.train.Coordinator()
      threads = pipeline.start(sess, coord)

      try:
        if top_k:
          with open('out_prob.txt', 'w+') as f:
            while not coord.should_stop():
              probs, cls_labels = sess.run([prob_op, labels])
              indices = np.argsort(-probs, axis=1) # descending
              for i in range(indices.shape[0]):
                f.write('true class: ' + cls_dict[cls_labels[i]] + '\n')
                for j in range(min(top_k, indices.shape[1])):
                  f.write(cls_dict[indices[i, j]] + '\t' + probs[i, j] + '\n')
                f.write('\n\n')
        else:
          true_count = 0
          while not coord.should_stop():
            true_count += np.sum(sess.run(top_k_op))
      except tf.errors.OutOfRangeError as e:
        coord.request_stop(e)

      coord.request_stop()
      coord.join(threads)

      print 'eval accuracy: %.3f' % (true_count / len(pipeline.videos))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('input_file')
  parser.add_argument('--ckpt_dir', required=True)
  parser.add_argument('--top_k')
  args = parser.parse_args()
  evaluate(args.input_file, args.ckpt_dir, args.top_k)
