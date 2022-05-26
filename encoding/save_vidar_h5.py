import numpy as np
import cv2
import os
import argparse
import h5py
import time

def RawToSpike(video_seq, h, w):
    video_seq = np.array(video_seq).astype(np.uint8)
    img_size = h*w
    img_num = len(video_seq)//(img_size//8)
    SpikeMatrix = np.zeros([img_num, h, w], np.uint8)
    pix_id = np.arange(0,h*w)
    pix_id = np.reshape(pix_id, (h, w))
    comparator = np.left_shift(1, np.mod(pix_id, 8))
    byte_id = pix_id // 8

    for img_id in np.arange(img_num):
        id_start = img_id*img_size//8
        id_end = id_start + img_size//8
        cur_info = video_seq[id_start:id_end]
        data = cur_info[byte_id]
        result = np.bitwise_and(data, comparator)
        if args.flipud:
            SpikeMatrix[img_id, :, :] = np.flipud((result == comparator))
        else:
            SpikeMatrix[img_id, :, :] = (result == comparator)

    return SpikeMatrix

def save_to_h5(SpikeMatrix):
    h5path = os.path.join(args.data_root, args.h5_name)
    print('save', h5path)
    f = h5py.File(h5path, 'w')
    f['raw_spike'] = SpikeMatrix
    f.close()


parser = argparse.ArgumentParser( )
parser.add_argument('-dr', '--data_root', type=str, default='/raid/lwhu/vidarflow/test/dice', help='Root path of the data')
parser.add_argument('-dn', '--data_name', type=str, default='test.dat', help='Name of the data file')
parser.add_argument('-hn', '--h5_name', type=str, default='test.h5', help='Name of the h5 file')
parser.add_argument('-fl', '--flipud', action='store_true', help='Flip the raw spike')
args = parser.parse_args()


if __name__ == '__main__':
    data_path = os.path.join(args.data_root, args.data_name)
    f = open(data_path, 'rb')
    video_seq = f.read()
    video_seq = np.frombuffer(video_seq, 'b')
    sp_mat = RawToSpike(video_seq, 500, 800)
    save_to_h5(sp_mat)
