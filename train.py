from __future__ import absolute_import
from __future__ import division
import os

import numpy as np
import tensorflow as tf
from inputs_new import *
from evaluate import evaluate

import i3d
from config import *

# build the model
def inference(rgb_inputs, flow_inputs):
  with tf.variable_scope('RGB'):
    rgb_model = i3d.InceptionI3d(
      NUM_CLASSES, spatial_squeeze=True, final_endpoint='Logits')
    rgb_logits, _ = rgb_model(
      rgb_inputs, is_training=True, dropout_keep_prob=DROPOUT_KEEP_PROB)
  with tf.variable_scope('Flow'):
    flow_model = i3d.InceptionI3d(
        NUM_CLASSES, spatial_squeeze=True, final_endpoint='Logits')
    flow_logits, _ = flow_model(
        flow_inputs, is_training=True, dropout_keep_prob=DROPOUT_KEEP_PROB)
  return rgb_logits, flow_logits

# restore the pretrained weights, except for the last layer
def restore():
  # rgb
  rgb_variable_map = {}
  for variable in tf.global_variables():
    if variable.name.split('/')[0] == 'RGB':
      if 'Logits' in variable.name: # skip the last layer
        continue
      rgb_variable_map[variable.name.replace(':0', '')] = variable
  rgb_saver = tf.train.Saver(var_list=rgb_variable_map, reshape=True)
  # flow
  flow_variable_map = {}
  for variable in tf.global_variables():
    if variable.name.split('/')[0] == 'Flow':
      if 'Logits' in variable.name: # skip the last layer
        continue
      flow_variable_map[variable.name.replace(':0', '')] = variable
  flow_saver = tf.train.Saver(var_list=flow_variable_map, reshape=True)
  return rgb_saver, flow_saver


def tower_inference(rgb_inputs, flow_inputs, labels):
  rgb_logits, flow_logits = inference(rgb_inputs, flow_inputs)
  model_logits = rgb_logits + flow_logits
  return tf.reduce_sum(
             tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=labels, logits=model_logits)), model_logits

def average_gradients(tower_grads):
  average_grads = []
  for grad_and_vars in zip(*tower_grads):
    # Note that each grad_and_vars looks like the following:
    # ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
    grads = []
    for g, _ in grad_and_vars:
      expanded_g = tf.expand_dims(g, 0)
      grads.append(expanded_g)

    grads_concat = tf.concat(grads, axis=0)
    grads_mean = tf.reduce_mean(grads_concat, axis=0)

    v = grad_and_vars[0][1]
    average_grads.append((grads_mean, v))
  return average_grads

if __name__ == '__main__':
  train_pipeline = InputPipeLine(TRAIN_DATA)
  val_pipeline = InputPipeLine(VAL_DATA)

  is_training = tf.placeholder(tf.bool)

  opt = tf.train.GradientDescentOptimizer(LR)

  tower_grads = []
  tower_losses = []
  tower_topk_counts = []
  
  with tf.variable_scope(tf.get_variable_scope()):
    for i in range(2):
      with tf.name_scope('tower_%d' % i):
        rgbs, flows, labels = tf.cond(is_training, lambda: train_pipeline.get_batch(train=True), lambda: val_pipeline.get_batch(train=False)) 
        with tf.device('/gpu:%d' % i):
          loss, logits = tower_inference(rgbs, flows, labels)
          topk_count = tf.reduce_sum(tf.cast(tf.nn.in_top_k(logits, labels, 1), tf.float32))
          tf.get_variable_scope().reuse_variables()
          grads = opt.compute_gradients(loss)
          tower_grads.append(grads)
          tower_losses.append(loss)
          tower_topk_counts.append(topk_count)
  
  true_count_op = tf.reduce_sum(tower_topk_counts)
  avg_loss = tf.reduce_mean(tower_losses)
  grads = average_gradients(tower_grads)
  train_op = opt.apply_gradients(grads)
  rgb_saver, flow_saver = restore()

  # saver for fine tuning
  if not os.path.exists(TMPDIR):
    os.mkdir(TMPDIR)
  saver = tf.train.Saver(max_to_keep=SAVER_MAX_TO_KEEP)
  ckpt_path = os.path.join(TMPDIR, 'ckpt')
  if not os.path.exists(ckpt_path):
    os.mkdir(ckpt_path)

  with tf.Session(config=tf.ConfigProto(allow_soft_placement=False, log_device_placement=False)) as sess:
    sess.run(tf.global_variables_initializer())

    ckpt = tf.train.get_checkpoint_state(ckpt_path)
    if ckpt and ckpt.model_checkpoint_path:
      print 'Restoring from:', ckpt.model_checkpoint_path
      saver.restore(sess, ckpt.all_model_checkpoint_paths[-1])
    else:
      print 'No checkpoint file found, restoring pretrained weights...'
      rgb_saver.restore(sess, CHECKPOINT_PATHS['rgb_imagenet'])
      flow_saver.restore(sess, CHECKPOINT_PATHS['flow_imagenet'])
      print 'Restore Complete.'

    train_coord, train_threads = train_pipeline.start(sess)
    val_coord, val_threads = val_pipeline.start(sess)

    summary_writer = tf.summary.FileWriter(TMPDIR, sess.graph)

    try:
      it = 0
      while it < MAX_ITER and not train_coord.should_stop() and not val_coord.should_stop():
        if it % DISPLAY_ITER == 0:
          _, loss_val = sess.run([train_op, avg_loss], {is_training: True})
          print 'step %d, loss = %.3f' % (it, loss_val)
          loss_summ = tf.Summary(value=[
            tf.Summary.Value(tag="train_loss", simple_value=loss_val)
          ])
          summary_writer.add_summary(loss_summ, it)
        if it % SAVE_ITER == 0:
          #   saver.save(sess, os.path.join(ckpt_path, 'model_ckpt'), it)
        if it % VAL_ITER == 0:
          true_count = 0
          for i in range(0, len(val_pipeline.videos), NUM_GPUS * BATCH_SIZE):
            true_count += sess.run(true_count_op, {is_training: False})
          acc = true_count / len(val_pipeline.videos)
          print 'val accuracy: %.3f' % acc
          acc_summ = tf.Summary(value=[
            tf.Summary.Value(tag="val_acc", simple_value=acc)
          ])
          summary_writer.add_summary(acc_summ, it)
        else:
          sess.run(train_op, {is_training: True})
        it += 1
    except (KeyboardInterrupt, tf.errors.OutOfRangeError) as e:
      saver.save(sess, os.path.join(ckpt_path, 'model_ckpt'), it)
      train_coord.request_stop(e)
      val_coord.request_stop(e)

    train_coord.request_stop()
    train_coord.join(train_threads)
    val_coord.request_stop()
    val_coord.join(val_threads)
