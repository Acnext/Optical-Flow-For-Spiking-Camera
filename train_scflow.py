import argparse
import time
import torch
import torch.backends.cudnn as cudnn
import torch.optim
import torchvision.transforms as transforms
import models
import datetime
from tensorboardX import SummaryWriter
import cv2
import os
import os.path as osp
import numpy as np
import random
import glob
from torch.utils.data import Dataset, DataLoader
from utils import *
from visulization_utils import *
import warnings
warnings.filterwarnings('ignore')


model_names = sorted(name for name in models.__dict__ if name.islower() and not name.startswith("__"))
parser = argparse.ArgumentParser()
parser.add_argument('-dr', '--data_root', type=str, default='/home/huliwen/vidar_data/train', help='root path of train datasets')
parser.add_argument('-tr', '--test_root', type=str, default='/home/huliwen/vidar_data/test', help='root path of test datasets')

parser.add_argument('-dt', '--dt', type=int, default=10, help='delta index between the input for flow')
parser.add_argument('-sd', '--savedir', type=str, default='./vidarflow', help='path for saving results')
parser.add_argument('-a', '--arch', default='scflow', choices=model_names, 
                    help='model architecture, overwritten if pretrained is specified: ' + ' | '.join(model_names))
parser.add_argument('--epochs', default=50, type=int, help='number of total epochs to run')
parser.add_argument('--epoch_size', type=int, default=0)
parser.add_argument('-b', '--batch-size', default=4, type=int, help='mini-batch size')
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float, help='initial learning rate')
parser.add_argument('-bn', '--batch_norm', default=False, type=bool, help='if use batch normlization during training')
parser.add_argument('--start-epoch', default=0, type=int, help='manual epoch number (useful on restarts)')
parser.add_argument('--solver', default='adam',choices=['adam','sgd'], help='solver algorithms')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum for sgd, alpha parameter for adam')
parser.add_argument('--beta', default=0.999, type=float, help='beta parameter for adam')
parser.add_argument('-j', '--workers', default=8, type=int, help='number of data loading workers')
parser.add_argument('--weight-decay', '--wd', default=4e-4, type=float, help='weight decay')
parser.add_argument('--bias-decay', default=0, type=float, help='bias decay')
parser.add_argument('--evaluate-interval', default=5, type=int, help='Evaluate every \'evaluate interval\' epochs ')
parser.add_argument('--print-freq', '-p', default=200, type=int, help='print frequency')
parser.add_argument('--pretrained', dest='pretrained', default=None, help='path to pre-trained model')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
parser.add_argument('--milestones', default=[5,10,20,30,40,50,70,90,110,130,150,170], nargs='*', help='epochs at which learning rate is divided by 2')
parser.add_argument('-vis', '--vis-path', default='./vis/sup', type=str, help='path to save flow visualization')
parser.add_argument('-vi', '--vis-interval', default=20, type=str, help='n_iter interval of flow visulization')
parser.add_argument('-mb', '--max-vis-batch', default=4, type=int, help='max visualization batch')
parser.add_argument('--w_scales', type=list, default='1111', help='switch for ph_loss in different pyramid levels')
parser.add_argument('--print-detail', '-pd', action='store_true')
parser.add_argument('--eval_root', '-er', default='eval_vis/barepwc')
parser.add_argument('--save_name', '-sn', default=None)
parser.add_argument('--decay', '-dc', type=float, default=0.7)
args = parser.parse_args()

n_iter = 0

args.milestones = [int(i) for i in args.milestones]
print(args.milestones)

eval_vis_path = args.eval_root + '_dt{:d}'.format(args.dt)
if not osp.exists(eval_vis_path):
    os.makedirs(eval_vis_path)

class Train_loading(Dataset):
    def __init__(self, transform=None):
        self.samples = self.collect_samples()
        print('Train len(self.samples)', len(self.samples))

    def collect_samples(self):
        scene_list = list(range(0, 100))
        samples = []
        for scene in scene_list:
            spike_dir = osp.join(args.data_root, str(scene), 'encoding25_dt{:d}'.format(args.dt))
            flowgt_dir = osp.join(args.data_root, str(scene), 'dt={:d}'.format(args.dt), 'motion_vector')
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

        y0 = np.random.randint(0, 20)
        seq1 = seq1[:, y0:y0+480, :]
        seq2 = seq2[:, y0:y0+480, :]
        flow = flow[y0:y0+480, :, :]
        return seq1, seq2, flow

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        seq1, seq2, flow = self._load_sample(self.samples[index])
        return seq1, seq2, flow


class Test_loading(Dataset):
    def __init__(self,  scene=None, transform=None):
        self.scene = scene
        self.samples = self.collect_samples()
        # print('scene:{:s}, dt{:d}, length:{:d}'.format(self.scene, args.dt, len(self.samples)))

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


def train(train_loader, model, optimizer, epoch, train_writer):
    global n_iter, args
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()


    # switch to train mode
    model.train()
    end = time.time()
    mini_batch_size_v = args.batch_size
    batch_size_v = 4

    loss_dict = {}
    loss_dict['w_scales'] = args.w_scales

    for ww, data in enumerate(train_loader, 0):
        # get the inputs
        seq1_raw, seq2_raw, flowgt_raw = data

        # compute output
        seq1_raw = seq1_raw.cuda().type(torch.cuda.FloatTensor)
        seq2_raw = seq2_raw.cuda().type(torch.cuda.FloatTensor)
        flowgt_raw = flowgt_raw.cuda().type(torch.cuda.FloatTensor).permute([0, 3, 1, 2])

        padder = InputPadder(seq1_raw.shape)
        seq1, seq2, flowgt = padder.pad(seq1_raw, seq2_raw, flowgt_raw)

        B, C, H, W = seq1.shape
        flow_init = torch.zeros([B, 2, H, W])
        with torch.no_grad():
            flow, model_res_dict = model(seq1, seq2, flow_init, dt=args.dt)
            flow_init = flow[0].clone().detach()
        flow, model_res_dict = model(seq1, seq2, flow_init, dt=args.dt)

        # compute loss
        loss, loss_res_dict = supervised_loss(flow, flowgt, loss_dict) 
        
        flow_mean = loss_res_dict['flow_mean']

        if n_iter % args.vis_interval == 0:
            outflow_img(flow, args.vis_path, name_prefix='flow', max_batch=args.max_vis_batch)
            outflow_img([flowgt], args.vis_path, name_prefix='flowgt', max_batch=args.max_vis_batch)

        # compute gradient and do optimization step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # record loss and EPE
        train_writer.add_scalar('total_loss', loss.item(), n_iter)
        train_writer.add_scalar('flow_mean', flow_mean.item(), n_iter)
        
        losses.update(loss.item())

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if mini_batch_size_v*ww % args.print_freq < mini_batch_size_v:
            cur_lr = optimizer.state_dict()['param_groups'][0]['lr']
            print('Epoch: [{0}][{1}/{2}]\t Time {3}\t Data {4}\t Loss {5}\t Flow mean {6}\t LR {7}'
                    .format(epoch, mini_batch_size_v*ww, mini_batch_size_v*len(train_loader), batch_time, data_time, losses, flow_mean, cur_lr))
        n_iter += 1

    return losses.avg



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

        if i % 10 == 0:
            pred_flow_vis = flow_to_img_scflow(pred_flow)
            flowgt_vis = flow_to_img_scflow(flowgt)

            pred_flow_vis_path = osp.join(scene_eval_vis_path, '{:03d}_pred.png'.format(i))
            flowgt_vis_path = osp.join(scene_eval_vis_path, '{:03d}_gt.png'.format(i))

            cv2.imwrite(pred_flow_vis_path, pred_flow_vis)
            cv2.imwrite(flowgt_vis_path, flowgt_vis)

        AEE = compute_aee(flowgt, pred_flow)
        
        AEE_sum += AEE
        eval_time_sum += eval_time

        iters += 1

        if args.print_detail:
            print('Scene: {:s}, Index {:04d}, AEE: {:6.4f}, Eval Time: {:6.4f}'.format(scene, i, AEE, eval_time))

    # print('-------------------------------------------------------')
    print('Scene: {:s}, Mean AEE: {:6.4f}, Mean Eval Time: {:6.4f}'.format(scene, AEE_sum / iters, eval_time_sum / iters))
    print('-------------------------------------------------------')

    return AEE_sum / iters


def main():
    global args, best_EPE, image_resize, event_interval, spiking_ts, device, sp_threshold
    timestamp1 = datetime.datetime.now().strftime("%m-%d")
    timestamp2 = datetime.datetime.now().strftime("%H%M%S")

    if args.save_name == None:
        save_folder_name = '{},{},{}epochs{},b{},lr{},{}'.format(
            args.arch,
            args.solver,
            args.epochs,
            ',epochSize'+str(args.epoch_size) if args.epoch_size > 0 else '',
            args.batch_size,
            args.lr,
            timestamp2)
    else:
        save_folder_name = '{},{},{}epochs{},b{},lr{},{},{}'.format(
            args.arch,
            args.solver,
            args.epochs,
            ',epochSize'+str(args.epoch_size) if args.epoch_size > 0 else '',
            args.batch_size,
            args.lr,
            timestamp2,
            args.save_name)
        
    save_root = osp.join(args.savedir, timestamp1)
    save_path = osp.join(save_root, save_folder_name)

    if not osp.exists(args.vis_path):
        os.makedirs(args.vis_path)


    if not args.evaluate:
        print('=> Everything will be saved to {}'.format(save_path))
        
        if not osp.exists(save_root):
            os.makedirs(save_root)
        if not osp.exists(save_path):
            os.makedirs(save_path)

    train_writer = SummaryWriter(osp.join(save_path,'train'))
    test_writer = SummaryWriter(osp.join(save_path,'test'))
    output_writers = []
    for i in range(3):
        output_writers.append(SummaryWriter(osp.join(save_path,'test',str(i))))

    # Data loading code
    co_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
    ])

    # Test_dataset: Waiting to write

    Test_dataset = Test_loading()
    test_loader = DataLoader(dataset=Test_dataset,
                              batch_size=1,
                              shuffle=False,
                              num_workers=args.workers)

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

    assert(args.solver in ['adam', 'sgd'])
    print('=> setting {} solver'.format(args.solver))
    param_groups = [{'params': model.module.bias_parameters(), 'weight_decay': args.bias_decay},
                    {'params': model.module.weight_parameters(), 'weight_decay': args.weight_decay}]
    if args.solver == 'adam':
        optimizer = torch.optim.Adam(param_groups, args.lr, betas=(args.momentum, args.beta))
    elif args.solver == 'sgd':
        optimizer = torch.optim.SGD(param_groups, args.lr, momentum=args.momentum)

    if args.evaluate:
        with torch.no_grad():
            best_EPE = validate(test_loader, model, -1, output_writers)
        return

    Train_dataset = Train_loading(transform=co_transform)
    train_loader = DataLoader(dataset=Train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=args.workers)

    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=0.7)

    for epoch in range(args.start_epoch, args.epochs):
        # scheduler.step()
        if (epoch+1) in args.milestones:
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * args.decay

        # train for one epoch
        train_loss = train(train_loader, model, optimizer, epoch, train_writer)

        is_best = False
        save_name = '{:s}_ckpt.pth.tar'.format(str(epoch+1))
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.module.state_dict(),
        }, is_best, save_path, save_name)

        # Test at every 5 epoch during training
        if (epoch + 1)%args.evaluate_interval == 0:
            # evaluate on validation set
            with torch.no_grad():
                # for scene in ['hand', 'ball']:
                for scene in os.listdir(args.test_root):
                    Test_dataset = Test_loading(scene=scene)
                    test_loader = DataLoader(dataset=Test_dataset,
                                            batch_size=1,
                                            shuffle=False,
                                            num_workers=args.workers)
                    EPE = validate(test_loader, model, scene)
                    model.train()

            test_writer.add_scalar('mean EPE', EPE, epoch)

if __name__ == '__main__':
    main()
