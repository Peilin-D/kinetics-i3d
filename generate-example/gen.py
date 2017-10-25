#!/usr/bin/env python
''' creates a sample for testing I3D. '''

'''
v_CricketShot_g04_c01_flow.gif
v_CricketShot_g04_c01_flow.npy
v_CricketShot_g04_c01_rgb.gif
v_CricketShot_g04_c01_rgb.npy
'''

import cv2
import numpy as np
import glob
import os

def readFlow(name):
    f = open(name, 'rb')
    header = f.read(4)
    if header.decode("utf-8") != 'PIEH':
        raise Exception('Flow file header does not contain PIEH')
    width = np.fromfile(f, np.int32, 1).squeeze()
    height = np.fromfile(f, np.int32, 1).squeeze()
    flow = np.fromfile(f, np.float32, width * height * 2).reshape((height, width, 2))
    return flow.astype(np.float32)

def main():
    ''' generate rgb.npy and flow.npy files from frames extracted '''

    video_name = 'v_CricketShot_g04_c01'
    resize_len = 256
    crop_w = 224
    crop_h = 224

    image_files = sorted(glob.glob('frames/[0-9][0-9][0-9][0-9][0-9][0-9].png'))
    use_deepflow = False
    generate_gif = True

    if use_deepflow:
        flow_files = sorted(glob.glob('frames/[0-9][0-9][0-9][0-9][0-9][0-9].flo'))
        assert len(image_files) == len(flow_files) + 1, "[Error] check images and flow files"

    rgb_cube = np.empty(
            [1, len(image_files), crop_h, crop_w, 3],
            dtype=np.float32)

    if generate_gif:
        import imageio
        images = []
        gif_kargs = { 'duration': 1./25 }

    for frame_num, image_file in enumerate(image_files):
        print "[Info] Reading image={}, frame_num={}...".format(image_file, frame_num)
        img = cv2.imread(image_file)

        # convert color
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # resize
        h, w = img.shape[:2]
        resize_ratio = float(resize_len) / min(h, w)
        img = cv2.resize(img, None, fx=resize_ratio, fy=resize_ratio)

        # crop center
        h, w = img.shape[:2]
        center_y = (h - crop_h)/2
        center_x = (w - crop_w)/2
        img = img[center_y:center_y+crop_h, center_x:center_x+crop_w, :]

        '''
        # pixel values rescale to [-1, 1]
        img = ((img - np.min(img)) / (np.max(img) - np.min(img)) - 0.5) * 2
        '''

        '''
        # per-channel re-scaling
        for c in range(img.shape[3]):
            tmp = img[:,:,c]
            img[:,:,c] = ((tmp - np.min(tmp)) / (np.max(tmp) - np.min(tmp)) - 0.5) * 2
        '''
        rgb_cube[0,frame_num,:,:,:] = img

        # generated animated gif
        if generate_gif and frame_num != len(image_files):
            images.append(img)

    # per-channel re-scaling
    for c in range(rgb_cube.shape[4]):
        tmp = rgb_cube[:,:,:,:,c]
        rgb_cube[:,:,:,:,c] = ((tmp - np.min(tmp)) / (np.max(tmp) - np.min(tmp)) - 0.5) * 2
    '''
    # per-cube re-scaling
    rgb_cube = ((rgb_cube - np.min(rgb_cube)) / (np.max(rgb_cube) - np.min(rgb_cube)) - 0.5) * 2
    '''

    # save rgb cube
    rgb_out_file = '{}_rgb.npy'.format(video_name)
    np.save(open(rgb_out_file, 'w'), rgb_cube[:,1:,:,:,:])

    if generate_gif:
        rgb_gif_file = '{}_rgb.gif'.format(video_name)
        imageio.mimsave(rgb_gif_file, images, 'GIF', **gif_kargs)

    if use_deepflow:
        # initialize flow
        flow_cube = np.empty(
                [1, len(flow_files), crop_h, crop_w, 2],
                dtype=np.float32)

        if generate_gif:
            images = []

        for frame_num, flow_file in enumerate(flow_files):
            print "[Info] Reading flow={}, frame_num={}...".format(flow_file, frame_num)
            flow = readFlow(flow_file)

            # resize
            h, w = flow.shape[:2]
            resize_ratio = float(resize_len) / min(h, w)
            flow = cv2.resize(flow, None, fx=resize_ratio, fy=resize_ratio)

            # crop center
            h, w = flow.shape[:2]
            center_y = (h - crop_h)/2
            center_x = (w - crop_w)/2
            flow = flow[center_y:center_y+crop_h, center_x:center_x+crop_w, :]

            # clipping at +-20
            #flow = np.clip(flow, -20, 20)
            flow[flow >= 20] = 20
            flow[flow <= -20] = -20
            max_val = lambda x: max(max(x.flatten()), abs(min(x.flatten())))

            flow = flow / 20
            flow_cube[0, frame_num,:,:,:] = flow

            # generated animated gif
            # clipping at +-20 and center image around 0.5
            if generate_gif:
                flow_3ch = np.ones([flow.shape[0], flow.shape[1], 3], dtype=np.uint8) * 128
                flow_3ch[:,:,:2] = (flow + 1) * 128.
                images.append(flow_3ch)

        '''
        # per-channel re-scaling
        for c in range(flow_cube.shape[4]):
            tmp = flow_cube[:,:,:,:,c]
            flow_cube[:,:,:,:,c] = ((tmp - np.min(tmp)) / (np.max(tmp) - np.min(tmp)) - 0.5) * 2
        '''

        '''
        tmp = flow_cube
        flow_cube = ((tmp - np.min(tmp)) / (np.max(tmp) - np.min(tmp)) - 0.5) * 2
        '''

    else:
        from cv2 import DualTVL1OpticalFlow_create as DualTVL1

        # initialize flow
        flow_cube = np.empty(
                [1, rgb_cube.shape[1]-1, crop_h, crop_w, 2],
                dtype=np.float32)

        if generate_gif:
            images = []

        TVL1 = DualTVL1()
        for frame_num in range(rgb_cube.shape[1]):
            print "[Info] frame_num={}...".format(frame_num)
            img = cv2.cvtColor(rgb_cube[0,frame_num,:,:,:], cv2.COLOR_RGB2GRAY)
            if frame_num == 0:
                prev_img = img
                continue
            flow = TVL1.calc(prev_img, img, None)
            assert(flow.dtype == np.float32)
            flow[flow >= 20] = 20
            flow[flow <= -20] = -20
            max_val = lambda x: max(max(x.flatten()), abs(min(x.flatten())))

            #flow = flow / max_val(flow)
            flow = flow / 20
            flow_cube[0, frame_num - 1,:,:,:] = flow

            prev_img = img

            # generated animated gif
            # clipping at +-20 and center image around 0.5
            if generate_gif:
                flow_3ch = np.ones([flow.shape[0], flow.shape[1], 3], dtype=np.uint8) * 128
                flow_3ch[:,:,:2] = (flow + 1) * 128.
                images.append(flow_3ch)

    # save flow cube
    flow_out_file = '{}_flow.npy'.format(video_name)
    np.save(open(flow_out_file, 'w'), flow_cube)

    if generate_gif:
        flow_gif_file = '{}_flow.gif'.format(video_name)
        imageio.mimsave(flow_gif_file, images, 'GIF', **gif_kargs)


if __name__ == '__main__':
    main()
