import tensorflow as tf
import numpy as np
import os
import threading
from config import *

"""
The input pipeline combines feed_dict and queues
enqueue using feed_dict inspired by http://ischlag.github.io/2016/11/07/tensorflow-input-pipeline-for-large-datasets/
"""

class InputPipeLine(object):
  def __init__(self, input_file_name, num_epochs=None):
    """
    input_file_name: a text file listing all input file path (absolute path).
    num_epochs: limits the number of epochs the pipeline will go over the inputs.
      set num_epochs=None to unlimit the number of epochs
      set num_epochs=i may result in a smaller number of examples in the last batch.
    """

    self.num_frames = NUM_FRAMES
    self.batch_size = BATCH_SIZE
    self.stride = FRAME_STRIDE
    self.num_epochs = num_epochs
    self._build_cls_dict()

    self.videos = []
    with open(input_file_name, 'r') as f:
      for path in f.readlines():
        path = path.strip()
        if path:
          self.videos.append(path.strip())

    # placeholders
    self.rgb = tf.placeholder(tf.string, shape=[self.num_frames])
    self.flow_x = tf.placeholder(tf.string, shape=[self.num_frames])
    self.flow_y = tf.placeholder(tf.string, shape=[self.num_frames])
    self.label = tf.placeholder(tf.int32)

    self.queue = tf.FIFOQueue(capacity=QUEUE_CAPACITY, dtypes=[tf.string, tf.string, tf.string, tf.int32], shapes=[[self.num_frames],[self.num_frames],[self.num_frames],[]])

  def _build_cls_dict(self):
    self.cls_dict = {}
    with open('ucfTrainTestlist/classInd.txt', 'r') as f:
      for line in f.readlines():
        line = line.strip()
        ind, cls_name = line.split(' ')
        self.cls_dict[cls_name.lower()] = int(ind) - 1

  def _enqueue(self, sess, enqueue_op):
    epoch = 0
    videos = self.videos
    while True:
      np.random.shuffle(videos) # random shuffle every epoch
      for video_path in videos:
        cls_name = video_path.split('_')[1]
        sorted_list = np.sort(os.listdir(video_path))
        imgs = [os.path.join(video_path, img) for img in sorted_list if img.startswith('img')]
        flow_xs = [os.path.join(video_path, flow) for flow in sorted_list if flow.startswith('flow_x')]
        flow_ys = [os.path.join(video_path, flow) for flow in sorted_list if flow.startswith('flow_y')]
        assert len(imgs) == len(flow_xs)
        assert len(imgs) == len(flow_ys)
        if self.num_frames <= len(imgs):
          begin = np.random.randint(0, len(imgs) - self.num_frames + 1)
        else:
          begin = 0
          ori_len = len(imgs)
          while len(imgs) < self.num_frames:
            for i in range(0, ori_len, self.stride):
              imgs.append(imgs[i])
              flow_xs.append(flow_xs[i])
              flow_ys.append(flow_ys[i])
              if len(imgs) == self.num_frames:
                break

        imgs_out = imgs[begin:begin + self.num_frames]
        flow_xs_out = flow_xs[begin:begin + self.num_frames]
        flow_ys_out = flow_ys[begin:begin + self.num_frames]
        sess.run(enqueue_op, {self.rgb: imgs_out, self.flow_x: flow_xs_out, self.flow_y: flow_ys_out, self.label: self.cls_dict[cls_name.lower()]})
      if self.num_epochs is not None:
        epoch += 1
        if epoch == self.num_epochs:
          break
    sess.run(self.queue.close(cancel_pending_enqueues=True))


  def start(self, sess, coord):
    enqueue_op = self.queue.enqueue([self.rgb, self.flow_x, self.flow_y, self.label])
    enqueue_thread = threading.Thread(target=self._enqueue, args=[sess, enqueue_op])
    enqueue_thread.daemon = True
    enqueue_thread.start()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    return threads

  def get_batch(self, train=True):
    """
    set train=True to get a batch for training:
      1. random flip left-right
      2. random crop
    set train=False to get a batch for evaluation(val/test):
      1. central crop (according to the paper)
    """
    item = self.queue.dequeue()

    rgb_frames = []
    flow_x_frames = []
    flow_y_frames = []
    for i in range(self.num_frames):
      rgb_frames.append(tf.image.decode_jpeg(tf.read_file(item[0][i]), channels=3))
      flow_x_frames.append(tf.image.decode_jpeg(tf.read_file(item[1][i]), channels=1))
      flow_y_frames.append(tf.image.decode_jpeg(tf.read_file(item[2][i]), channels=1))

    output_rgb = tf.stack(rgb_frames, axis=0)
    tmp_flow_x = tf.stack(flow_x_frames, axis=0)
    tmp_flow_y = tf.stack(flow_y_frames, axis=0)
    output_flow = tf.concat([tmp_flow_x, tmp_flow_y], axis=3)

    rgb_flow_concat = tf.concat([output_rgb, output_flow], axis=3)
    if train:
      # random flip left-right
      rand_num = tf.random_uniform([])
      flip_concat = tf.cond(tf.less(rand_num, 0.5), lambda: tf.reverse(rgb_flow_concat, axis=[2]), lambda: rgb_flow_concat)
      # random crop
      crop_concat = tf.random_crop(flip_concat, [int(self.num_frames), CROP_SIZE, CROP_SIZE, 5])
    else:
      # center crop
      beginH = (tf.shape(rgb_flow_concat)[1] - 224) / 2
      beginW = (tf.shape(rgb_flow_concat)[2] - 224) / 2
      crop_concat = tf.slice(rgb_flow_concat, [0, beginH, beginW, 0], [-1, CROP_SIZE, CROP_SIZE, -1])

    output_rgb = crop_concat[:,:,:,:3]
    output_flow = crop_concat[:,:,:,3:]

    # rescale
    output_rgb = tf.cast(output_rgb, tf.float32)
    output_flow = tf.cast(output_flow, tf.float32)
    output_rgb = output_rgb * 2 / 255.0 - 1
    output_flow = output_flow * 2 / 256.0 - 1

    label = tf.cast(item[3], tf.int32)
    rgbs, flows, labels = tf.train.batch([output_rgb, output_flow, label], batch_size=self.batch_size, allow_smaller_final_batch=True)
    return rgbs, flows, labels

  def prefetch_queue(self):
    queue = tf.PaddingFIFOQueue(capacity=2,
                                dtypes=[tf.float32, tf.float32, tf.int32], 
                                shapes=[[None, NUM_FRAMES, CROP_SIZE, CROP_SIZE, 3],
                                        [None, NUM_FRAMES, CROP_SIZE, CROP_SIZE, 2],
                                        [None]])
    batch_data = self.get_batch()
    enq = queue.enqueue(batch_data)
    tf.train.add_queue_runner(tf.train.QueueRunner(queue, [enq]))
    return queue

