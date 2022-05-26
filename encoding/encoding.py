import numpy as np
import os
import h5py
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('-dr', '--data_root', type=str, default='/raid/lwhu/vidarflow/test/dice', help='root path of datasets')
parser.add_argument('-dn', '--data_name', type=str, default='test.h5', help='name of raw files')
parser.add_argument('-sn', '--save_dir', type=str, default='encoding', help='name of saved files')
parser.add_argument('-dt', '--dt', type=int, default=10, help='dt')
parser.add_argument('-l', '--data_length', type=int, default=11, help='length of spike sequence for each group')
args = parser.parse_args()


if __name__ == '__main__':
    data_path = os.path.join(args.data_root, args.data_name)
    save_path = os.path.join(args.data_root, args.save_dir+'_dt'+str(args.dt))

    if not os.path.exists(save_path):
        os.makedirs(save_path)
  
    f1 = h5py.File(data_path, 'r')
    raw_spike = f1['raw_spike']
    c, h, w = raw_spike.shape
    data_step = args.dt
    half_length = (args.data_length-1) // 2

    ii = 0
    while True:
        central_index = ii * data_step
        st_index = central_index - half_length
        ed_index = central_index + half_length + 1


        if (ed_index >= c - 40):
            break

        if (central_index < 40):
            ii += 1
            continue

        np.save(os.path.join(save_path, str(ii)), raw_spike[st_index:ed_index, :, :])
        print('Finish process {:s} #{:04d} sample : rep1 length={:02d} dt{:d}'.format(args.data_root, ii, args.data_length, args.dt))
        
        ii += 1