import argparse
import time
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import models
import cv2
import os
import os.path as osp
import numpy as np
import glob
from torch.utils.data import Dataset, DataLoader
from utils import *
from visulization_utils import *

import warnings
warnings.filterwarnings('ignore')

###############################################################################################################################################


model_names = sorted(name for name in models.__dict__ if name.islower() and not name.startswith("__"))
parser = argparse.ArgumentParser()
parser.add_argument('-tr', '--test_root', type=str, default='/home/huliwen/vidar_data/test', help='root path of test datasets')
parser.add_argument('-dt', '--dt', type=int, default=10, help='delta index between the input for flow')
parser.add_argument('-a', '--arch', default='scflow', choices=model_names, 
                    help='model architecture, overwritten if pretrained is specified: ' + ' | '.join(model_names))
parser.add_argument('-bn', '--batch_norm', default=False, type=bool, help='if use batch normlization during training')
parser.add_argument('-j', '--workers', default=8, type=int, help='number of data loading workers')
parser.add_argument('--pretrained', dest='pretrained', default=None, help='path to pre-trained model')
parser.add_argument('--print-detail', '-pd', action='store_true')
parser.add_argument('--eval_root', '-er', default='eval_vis_all/scflow')
args = parser.parse_args()

n_iter = 0
eval_vis_path = args.eval_root + '_dt{:d}'.format(args.dt)
if not osp.exists(eval_vis_path):
    os.makedirs(eval_vis_path)

class Test_loading(Dataset):
    def __init__(self,  scene=None, transform=None):
        self.scene = scene
        self.samples = self.collect_samples()

    def collect_samples(self):
        scene_list = [self.scene]
        samples = []
        for scene in scene_list:
            spike_dir = osp.join(args.test_root, str(scene), 'encoding25_dt{:d}'.format(args.dt))
            flowgt_dir = osp.join(args.test_root, str(scene), 'dt={:d}'.format(args.dt), 'motion_vector')
            for st in range(0, len(glob.glob(spike_dir+'/*.npy')) - 1):
                seq1_path  = spike_dir + '/' + str(int(st)) + '.npy'
                seq2_path  = spike_dir + '/' + str(int(st+1)) + '.npy'
                flow_path = flowgt_dir + '/{:04d}.flo'.format(int(st))
                if osp.exists(seq1_path) and osp.exists(seq2_path) and osp.exists(flow_path):
                    s = {}
                    s['seq1_path'], s['seq2_path'], s['flow_path'] = seq1_path, seq2_path, flow_path
                    samples.append(s)
        return samples

    def _load_sample(self, s):
        seq1 = np.load(s['seq1_path'], allow_pickle=True).astype(np.float32)
        seq2 = np.load(s['seq2_path'], allow_pickle=True).astype(np.float32)
        flow = readFlow(s['flow_path']).astype(np.float32)
        return seq1, seq2, flow

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        seq1, seq2, flow = self._load_sample(self.samples[index])
        return seq1, seq2, flow


def validate(test_loader, model, scene):
    model.eval()

    AEE_sum = 0.
    eval_time_sum = 0.
    iters = 0.
    scene_eval_vis_path = osp.join(eval_vis_path, scene)
    if not osp.exists(scene_eval_vis_path):
        os.makedirs(scene_eval_vis_path)

    for i, data in enumerate(test_loader, 0):
        seq1_raw, seq2_raw, flowgt_raw = data

        # compute output
        seq1_raw = seq1_raw.cuda().type(torch.cuda.FloatTensor)
        seq2_raw = seq2_raw.cuda().type(torch.cuda.FloatTensor)
        flowgt = flowgt_raw.cuda().type(torch.cuda.FloatTensor).permute([0, 3, 1, 2])

        padder = InputPadder(seq1_raw.shape)
        seq1, seq2 = padder.pad(seq1_raw, seq2_raw)

        st_time = time.time()
        if i == 0:
            B, C, H, W = seq1.shape
            flow_init = torch.zeros([B, 2, H, W])
        with torch.no_grad():
            flows, model_res_dict = model(seq1=seq1, seq2=seq2, flow=flow_init, dt=args.dt)
        eval_time = time.time() - st_time

        flow_init = flows[0].clone().detach()
        flow_init = flow_warp(flow_init, -flow_init)

        pred_flow = padder.unpad(flows[0]).detach().permute([0, 2, 3, 1]).squeeze().cpu().numpy()
        flowgt = flowgt.detach().permute([0, 2, 3, 1]).squeeze().cpu().numpy()

        pred_flow_vis = flow_to_img_scflow(pred_flow)
        pred_flow_vis_path = osp.join(scene_eval_vis_path, '{:03d}_pred.png'.format(i))
        cv2.imwrite(pred_flow_vis_path, pred_flow_vis)

        AEE = compute_aee(flowgt, pred_flow, thresh=1)
        
        AEE_sum += AEE
        eval_time_sum += eval_time

        iters += 1

        if args.print_detail:
            print('Scene: {:8s}, Index {:04d}, AEE: {:6.4f}, Eval Time: {:6.4f}'.format(scene, i, AEE, eval_time))
            print(percent_AEE)

    # print('-------------------------------------------------------')
    print('Scene: {:s}, Mean AEE: {:6.4f}, Mean Eval Time: {:6.4f}'.format(scene, AEE_sum / iters, eval_time_sum / iters))
    print(percent_AEE_sum/iters)
    print('-------------------------------------------------------')

    return AEE_sum / iters


def main():
    global args, best_EPE, image_resize, event_interval, spiking_ts, device, sp_threshold

    # create model
    if args.pretrained:
        network_data = torch.load(args.pretrained)
        print("=> using pre-trained model '{}'".format(args.arch))
    else:
        network_data = None
        print("=> creating model '{}'".format(args.arch))
    
    model = models.__dict__[args.arch](network_data, args.batch_norm).cuda()
    model = torch.nn.DataParallel(model).cuda()
    cudnn.benchmark = True

    # for scene in os.listdir(args.test_root):
    for scene in ['ball', 'cook', 'dice', 'dolldrop', 'fan', 'fly', 'hand', 'jump', 'poker', 'top']:
        Test_dataset = Test_loading(scene=scene)
        test_loader = DataLoader(dataset=Test_dataset,
                                batch_size=1,
                                shuffle=False,
                                num_workers=args.workers)
        EPE = validate(test_loader, model, scene)
        model.train()

if __name__ == '__main__':
    main()
