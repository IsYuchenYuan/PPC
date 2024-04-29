import argparse
import logging
import os
import random
import sys

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import wandb
from einops import rearrange

from dataloaders.dataset import (BaseDataSets, RandomGenerator, TwoStreamBatchSampler)
from networks.unet_proto import UNetProto
from utils import losses, ramps, val_2d, util
from scheduler.my_lr_scheduler import PolyLR

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='../data/ACDC', help='Name of Experiment')
parser.add_argument('--exp', type=str, default='softplus', help='experiment_name')
parser.add_argument('--proj', type=str, default='ACDC', help='wandb project name')
parser.add_argument('--phase', type=str, default='unify', help='training phase')
parser.add_argument('--model', type=str, default='unet', help='model_name')

parser.add_argument('--max_iterations', type=int, default=30000, help='maximum epoch number to train')
parser.add_argument("--pretrainIter", type=int, default=6000, help="maximum iteration to pretrain")
parser.add_argument("--linearIter", type=int, default=1000, help="maximum iteration to pretrain")

parser.add_argument('--sim_mode', type=str, default='dist', help='similarity computation method [dist,euclidean]')
parser.add_argument('--embed_dim', type=int, default=128, help='the dimension of the mu')
parser.add_argument('--sigma_mode', type=str, default='diagonal',
                    help='the type of covariance matrix [radius,diagonal]')
parser.add_argument('--sigma_trans_mode', type=str, default='sigmoid',
                    help='the way to transform sigma_raw to sigma')  # softplus, sigmoid, sigmoidLearn
parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
parser.add_argument('--base_lr', type=float, default=1e-2, help='segmentation network learning rate')  # 1e-2 8e-3
parser.add_argument('--uncer_lr', type=float, default=1e-2, help='uncertainty network learning rate')
parser.add_argument('--min_lr', type=float, default=1e-7, help='minmum lr the scheduler to reach')  # le-6
parser.add_argument('--patch_size', type=list, default=[256, 256], help='patch size of network input')
parser.add_argument('--seed', type=int, default=1337, help='random seed')
parser.add_argument('--num_classes', type=int, default=4, help='output channel of network')
# label and unlabel
parser.add_argument('--batch_size', type=int, default=12, help='batch_size per gpu')
parser.add_argument('--labeled_bs', type=int, default=6, help='labeled_batch_size per gpu')  # 6
parser.add_argument('--labelnum', type=int, default=7, help='labeled data')
# costs
parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
parser.add_argument('--ema_decay', type=float, default=0.99, help='ema_decay')
parser.add_argument('--consistency', type=float, default=2, help='consistency')
parser.add_argument('--constant', type=bool, default=False, help='whether to use constant un weight')
parser.add_argument('--consistency_rampup', type=float, default=240.0, help='consistency_rampup')

parser.add_argument("--pretrain", type=bool, default=False, help="whether to initialize prototype")
parser.add_argument("--dice_w", type=float, default=0.5, help="the weight of dice loss")
parser.add_argument("--ce_w", type=float, default=0.5, help="the weight of ce loss")
parser.add_argument("--proto_w", type=float, default=1, help="the weight of proto loss")
parser.add_argument('--proto_update_interval', type=int, default=200, help='the interval iterations for proto updating')
parser.add_argument("--multiDSC", type=bool, default=True, help="whether use multiDSC")
parser.add_argument("--losstype", type=str, default="ce_dice", help="the type of ce and dice loss")
parser.add_argument('--momentum', type=bool, default=False, help='whether use momentum to update protos')

# temperature for softmax in proto matching
parser.add_argument('--temp', type=float, default=0.5, help='temperature for softmax in proto matching')
parser.add_argument('--thre_u1', type=float, default=90, help='threshold for unlabeled loss in the first 1k iterations')
parser.add_argument('--thre_u2', type=float, default=80,
                    help='threshold for unlabeled loss in the remaining iterations')
parser.add_argument('--thre_weak', type=float, default=80, help='weak threshold for sampling the representations')
parser.add_argument('--thre_strong', type=float, default=10,
                    help='strong threshold for sampling the representations')  # in percentile
parser.add_argument('--sample', type=str, default='reliability', help='the type of sampling')  # reliability/coreset
parser.add_argument('--coresize', type=int, default=5000, help='the size of coreset')

parser.add_argument('--device', type=str, help='the device')
parser.add_argument('--wandb', type=bool, default=True, help='whether use wandb')

args = parser.parse_args()

num_classes = args.num_classes

os.environ["CUDA_VISIBLE_DEVICES"] = args.device

if not args.wandb:
    # If you don't want your script to sync to the cloud
    os.environ['WANDB_MODE'] = 'dryrun'


# Method 1: Reliability
def cal_sampling_reliability_mask(label, sigma_sq):
    """

    :param sigma_sq: (B,C,H,W)
    :param label: (B,H,W)
    :return:
    """
    B = label.size(0)
    sigma_channel_last = sigma_sq.permute(0, 2, 3, 1)  # b h w c
    sample_map = torch.zeros_like(label).float()
    for idx in range(num_classes):
        mask_p_c = torch.zeros_like(label).float()
        if idx != 0:
            mask_p_c = (label == idx).float()
        else:
            sigma_mean = torch.mean(sigma_channel_last, dim=-1)  # b h w
            mask = label == idx # b h w
            sigma_mean_c = sigma_mean[mask.bool()]
            sigma_mean_c_np = sigma_mean_c.detach().cpu().numpy()
            for i in range(B):
                if torch.sum(mask[i]) == 0:
                    continue
                mask_p_c[i] = ((sigma_mean[i].le(np.percentile(sigma_mean_c_np, args.thre_weak))
                                & sigma_mean[i].ge(np.percentile(sigma_mean_c_np,
                                                                 args.thre_strong))).float()) * mask[i]  # choosing representations to update prototypes
        sample_map += mask_p_c

    return sample_map


# Method 2: Coreset
def cal_sampling_mask_coreset_fixNum(label, mu, num_list):
    """
    :param label: (B,H,W)
    :param mu: (B,C,H,W)
    :param num_list: the number of coreset for each class. list
    :return:
    """
    mu_channel_last = mu.permute(0, 2, 3, 1)  # b h w c
    if torch.isnan(mu_channel_last).any():
        print("the model break down, contains NaN !!!!!!!!!")
        assert 1==0
    b, h, w = label.size()
    sample_map = torch.zeros_like(label).float().cuda()
    for idx in range(num_classes):
        num = num_list[idx]
        mask = label == idx
        sample_map_c = torch.zeros_like(label).float().cuda() # b h w
        for i in range(b):
            # print("{}th batch {} class number: {}".format(i,idx, torch.sum(mask[i])))
            if torch.sum(mask[i]) <= num:
                sample_map_c[i] = mask[i]
                continue
            mu_c_mean = torch.mean(mu_channel_last[i][mask[i].bool()], dim=0).unsqueeze(0)  # 1,c
            # dist_sum = torch.sum(1- F.cosine_similarity(mu_channel_last[mask.bool()],mu_c_mean,dim=-1))
            dist_all = 1 - F.cosine_similarity(mu_channel_last[i], mu_c_mean, dim=-1)  # find the most dominant features
            dist_all = torch.clamp(dist_all, 0, 2)
            dist_c = dist_all * mask[i]
            qx_c = 0.5 * 1 / torch.sum(mask[i]) + 0.5 * dist_c / (torch.max(dist_c) + 1e-7)  # b h w d
            qx_c = qx_c * mask[i]
            p = rearrange(qx_c, "h w->  (h w)").detach().cpu().numpy()
            p = p / np.linalg.norm(p, ord=1)
            sample_map_flatten = rearrange(sample_map_c[i], "h w ->  (h w)")
            sample_map_flatten_cpu = sample_map_flatten.detach().cpu().numpy()
            sample_idx = np.random.choice(np.arange(sample_map_flatten_cpu.shape[0]), num, p=p,replace=False)
            sample_map_flatten_cpu[sample_idx] = 1
            sample_map_c[i] = rearrange(torch.from_numpy(sample_map_flatten_cpu).cuda(), "(h w) -> h w ", h=h)
        sample_map += sample_map_c
    return sample_map


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)

    ema_model.prototypes_mu.mul_(alpha).add_(1 - alpha, model.prototypes_mu)
    ema_model.prototypes_sigma.mul_(alpha).add_(1 - alpha, model.prototypes_sigma)

    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


def get_current_proto_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.proto_w * ramps.sigmoid_rampup(epoch, args.proto_rampup)


###############loss##########
weights = torch.from_numpy(np.asarray([0.05, 0.35, 0.35, 0.25])).float().cuda()

if args.losstype == "wce_dice":
    CE = torch.nn.CrossEntropyLoss(weight=weights)
    dice_loss = losses.DiceLoss(num_classes, weights=None)
elif args.losstype == "wce_wdice":
    CE = torch.nn.CrossEntropyLoss(weight=weights)
    dice_loss = losses.DiceLoss(num_classes, weights=weights)
elif args.losstype == "ce_dice":
    CE = torch.nn.CrossEntropyLoss(weight=None)
    dice_loss = losses.DiceLoss(num_classes, weights=None)


def mix_loss(output, label):
    # LogSumExp -> softmax(x) = softmax(x-b) b is the largest num in x
    # output = output - torch.max(output,dim=1,keepdim=True)[0]

    loss_ce = CE(output, label)
    if args.multiDSC:
        output_soft = F.softmax(output, dim=1)
        loss_dice = dice_loss(output_soft, label)
    else:
        loss_dice = losses.dice_loss_foreground(output, label)
    return loss_ce, loss_dice

def unsup_mix_loss(output, label, weight):
    # LogSumExp -> softmax(x) = softmax(x-b) b is the largest num in x
    # output = output - torch.max(output,dim=1,keepdim=True)[0]

    loss_ce = (F.cross_entropy(output, label,
                               reduction='none') * weight).mean()
    if args.multiDSC:
        output_soft = F.softmax(output, dim=1)
        loss_dice = dice_loss(output_soft, label, mask=weight)
    else:
        loss_dice = losses.dice_loss_foreground(output, label)
    return loss_ce, loss_dice


def patients_to_slices(dataset, patiens_num):
    ref_dict = None
    if "ACDC" in dataset:
        ref_dict = {"1": 32, "3": 68, "7": 136,
                    "14": 256, "21": 396, "28": 512, "35": 664, "70": 1312}
    elif "Prostate":
        ref_dict = {"2": 27, "4": 53, "8": 120,
                    "12": 179, "16": 256, "21": 312, "42": 623}
    else:
        print("Error")
    return ref_dict[str(patiens_num)]


def train(args, snapshot_path):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    labeled_bs, unlabeled_bs = args.labeled_bs, args.batch_size - args.labeled_bs

    pretrain = False
    if args.pretrain:
        pretrain = True

    checkpoint = "../model/ACDC/7_labeled/ACDC_dim128_unet_6kpretrain_weight2_7_labeled/unify/unet_best_model.pth"
    # checkpoint = "../model/ACDC/7_labeled/samplebg_strong10_weak80_dim128_6kpretrain_weight2_polylr_bs6_dist_diagonal_sigmoid_temp100_uncerlr0.01/unify/unet_best_model.pth"
    checkpoint_s_mu = '../model/LAsingle/4_labeled/inverse_softplus_strong10_weak80_dim32_3kpretrain_weight2_polylr_bs1_dist_diagonal_sigmoid_temp0.5_uncerlr0.01/' \
                      'unify/proto_mu.pt'
    checkpoint_s_sigma = '../model/LAsingle/4_labeled/inverse_softplus_strong10_weak80_dim32_3kpretrain_weight2_polylr_bs1_dist_diagonal_sigmoid_temp0.5_uncerlr0.01/' \
                         'unify/proto_sigma.pt'
    checkpoint_t_mu = ''
    checkpoint_t_sigma = ''

    base_lr = args.base_lr
    uncer_lr = args.uncer_lr
    iter_num = 0
    best_performance = 0.0
    bestIter = 0
    start_epoch = 0

    def create_model(pretrain=False, ema=False):
        net = UNetProto(
            backbone=args.model,
            inchannel=1,
            nclasses=num_classes,
            proto_mu=None,
            proto_sigma=None,
            embed_dim=args.embed_dim,
            sigma_mode=args.sigma_mode,
            sigma_trans_mode=args.sigma_trans_mode,
            sim_mode=args.sim_mode,
            temp=args.temp,
            momentum=args.momentum,
        )
        net = net.cuda()
        if pretrain:
            net.load_state_dict(torch.load(checkpoint))
            # sigma = net.prototypes_sigma
            # mu = net.prototypes_mu
            # print(1/sigma.mean(-1))
            # print(mu.mean(-1))
            # assert 1==0
            if ema:
                net.prototypes_mu = torch.load(checkpoint_t_mu)
                net.prototypes_sigma = torch.load(checkpoint_t_sigma)
            else:
                net.prototypes_mu = torch.load(checkpoint_s_mu)
                net.prototypes_sigma = torch.load(checkpoint_s_sigma)
        if ema:
            for param in net.parameters():
                param.detach_()
        return net

    s_model = create_model(pretrain=pretrain, ema=False)
    t_model = create_model(pretrain=pretrain, ema=True)

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    db_train = BaseDataSets(base_dir=args.root_path,
                            split="train",
                            num=None,
                            transform=transforms.Compose([RandomGenerator(args.patch_size)]))
    db_val = BaseDataSets(base_dir=args.root_path, split="val")
    total_slices = len(db_train)
    labeled_slice = patients_to_slices(args.root_path, args.labelnum)
    print("Total slices is: {}, labeled slices is:{}".format(total_slices, labeled_slice))
    labeled_idxs = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, args.batch_size,
                                          args.batch_size - args.labeled_bs)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=4, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=1)

    max_epoch = args.max_iterations // len(trainloader) + 1

    optimizer = optim.SGD(s_model.backbone.parameters(), lr=base_lr,
                          momentum=0.9, weight_decay=0.0001, nesterov=True)
    optimizer_uncer = optim.SGD(s_model.uncer_head.parameters(), lr=uncer_lr,
                                weight_decay=0.0001,
                                momentum=0.9, nesterov=True)

    lr_scheduler_uncer = PolyLR(optimizer_uncer, max_epoch, min_lr=args.min_lr)
    lr_scheduler = PolyLR(optimizer, max_epoch, min_lr=args.min_lr)

    wandb.init(project=args.proj,
               name=args.exp,
               entity='isyuanyc')
    logging.info("Start self_training")
    logging.info("{} iterations per epoch".format(len(trainloader)))

    s_model.train()
    t_model.train()

    if pretrain:
        start_epoch = iter_num // len(trainloader) + 1
    iterator = tqdm(range(start_epoch, max_epoch), ncols=70)
    for epoch in iterator:
        for _, sampled_batch in enumerate(trainloader):

            # torch.autograd.set_detect_anomaly(True)
            base_lr_ = optimizer.param_groups[0]['lr']
            uncer_lr_ = optimizer_uncer.param_groups[0]['lr']
            iter_num = iter_num + 1

            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

            volume_batch_l, label_batch_l = volume_batch[:labeled_bs], label_batch[:labeled_bs]
            volume_batch_ul, label_batch_ul = volume_batch[labeled_bs:], label_batch[labeled_bs:]

            log_dict = {'iterNum': iter_num, 'lr': base_lr_, 'uncer_lr': uncer_lr_}
            log_dict['proto_mu_0'] = torch.mean(s_model.prototypes_mu[0])
            log_dict['proto_mu_1'] = torch.mean(s_model.prototypes_mu[1])
            log_dict['proto_mu_2'] = torch.mean(s_model.prototypes_mu[2])
            log_dict['proto_mu_3'] = torch.mean(s_model.prototypes_mu[3])
            log_dict['proto_sigma_0'] = torch.mean(s_model.prototypes_sigma[0])
            log_dict['proto_sigma_1'] = torch.mean(s_model.prototypes_sigma[1])
            log_dict['proto_sigma_2'] = torch.mean(s_model.prototypes_sigma[2])
            log_dict['proto_sigma_3'] = torch.mean(s_model.prototypes_sigma[3])

            if iter_num <= args.linearIter:
                cls_seg = s_model.warm_up(x_2d=volume_batch_l)
                loss_cls_ce, loss_seg_dice = mix_loss(cls_seg, label_batch_l)
                loss_cls_2d = args.ce_w * loss_cls_ce + args.dice_w * loss_seg_dice
                loss = loss_cls_2d
                logging.info('training linear cls only ... iteration %d : avg loss : %f  '
                             % (iter_num, loss.item()))
            elif iter_num <= args.pretrainIter:
                outputs = s_model(x_2d=volume_batch_l, label=label_batch_l)

                cls_seg = outputs["cls_seg"]  # b,c,h,w
                loss_cls_ce, loss_seg_dice = mix_loss(cls_seg, label_batch_l)
                loss_cls_2d = args.ce_w * loss_cls_ce + args.dice_w * loss_seg_dice

                proto_seg = outputs["proto_seg"]
                mu = outputs["mu"]  # b c h w
                sigma_sq = outputs["sigma"]  # b c h w

                a_soft = F.softmax(proto_seg, dim=1)
                a_soft_max = torch.max(a_soft, dim=1)[0].detach().clone().cpu().numpy()
                a_soft_max_cpu_30 = np.percentile(a_soft_max, 30)
                a_soft_max_cpu_60 = np.percentile(a_soft_max, 60)
                a_soft_max_cpu_85 = np.percentile(a_soft_max, 85)
                a_soft_max_cpu_95 = np.percentile(a_soft_max, 95)

                log_dict['proto_soft_max_30'] = a_soft_max_cpu_30
                log_dict['proto_soft_max_60'] = a_soft_max_cpu_60
                log_dict['proto_soft_max_85'] = a_soft_max_cpu_85
                log_dict['proto_soft_max_95'] = a_soft_max_cpu_95

                # log the sigma--------------------------------------------
                sigma = torch.mean(sigma_sq, dim=1)
                sigma_cpu = sigma.clone().detach().cpu().numpy()  # b h w
                sigma_cpu_30 = np.percentile(sigma_cpu, 30)
                sigma_cpu_60 = np.percentile(sigma_cpu, 60)
                sigma_cpu_85 = np.percentile(sigma_cpu, 85)
                sigma_cpu_95 = np.percentile(sigma_cpu, 95)
                log_dict['l_sigma_30'] = sigma_cpu_30
                log_dict['l_sigma_60'] = sigma_cpu_60
                log_dict['l_sigma_85'] = sigma_cpu_85
                log_dict['l_sigma_95'] = sigma_cpu_95
                # ----------------------------------------------------

                # log mu by class--------------------------------------------
                # mu_mean = torch.mean(mu, dim=1)
                # mu_mean = rearrange(mu_mean, "(b d) h w -> b h w d ", b=volume_batch_l.size(0))
                #
                # mu_0 = mu_mean[label_batch_l == 0]
                # mu_1 = mu_mean[label_batch_l == 1]
                # mu_2 = mu_mean[label_batch_l == 2]
                # mu_3 = mu_mean[label_batch_l == 3]
                #
                # mu_0_cpu = mu_0.detach().cpu().numpy()  # b h w d
                # mu_1_cpu = mu_1.detach().cpu().numpy()  # b h w d
                # mu_2_cpu = mu_2.detach().cpu().numpy()  # b h w d
                # mu_3_cpu = mu_3.detach().cpu().numpy()  # b h w d
                # mu_0_cpu_95 = np.percentile(mu_0_cpu, 95)
                # log_dict['mu0_95'] = mu_0_cpu_95
                # mu_1_cpu_95 = np.percentile(mu_1_cpu, 95)
                # log_dict['mu1_95'] = mu_1_cpu_95
                # mu_2_cpu_95 = np.percentile(mu_2_cpu, 95)
                # log_dict['mu2_95'] = mu_2_cpu_95
                # mu_3_cpu_95 = np.percentile(mu_3_cpu, 95)
                # log_dict['mu3_95'] = mu_3_cpu_95
                # ----------------------------------------------------

                if args.sample == 'reliability':
                    mask_p = cal_sampling_reliability_mask(label=label_batch_l, sigma_sq=sigma_sq)
                elif args.sample == 'coreset':
                    mask_p = cal_sampling_mask_coreset_fixNum(label=label_batch_l, mu=mu,
                                                              num_list=[args.coresize for i in range(num_classes)])

                # update the prototypes
                if iter_num % args.proto_update_interval ==0:
                    s_model.prototype_update(mu, sigma_sq, label_batch_l, mask_p)

                loss_proto_ce, loss_proto_dice = mix_loss(proto_seg, label_batch_l)
                loss_proto_2d = args.ce_w * loss_proto_ce + args.dice_w * loss_proto_dice
                loss = loss_cls_2d + args.proto_w * loss_proto_2d
                logging.info('train both cls ... iteration %d : lr: %f uncer_lr : %f avg loss : %f  loss_cls_2d : %f  '
                             'loss_proto : %f'
                             % (iter_num, base_lr_, uncer_lr_, loss.item(), loss_cls_2d.item(),
                                loss_proto_2d.item()))

            elif iter_num > args.pretrainIter:
                with torch.no_grad():
                    u_output = t_model(x_2d=volume_batch_ul)
                    # u_cls_seg = u_output["cls_seg"]
                    u_cls_seg_p = u_output["proto_seg"]
                    prob = F.softmax(u_cls_seg_p, dim=1)
                    sigma = u_output["sigma"]  # B, dim, H, W
                    mu = u_output["mu"]  # B, dim, H, W

                    max_probs, max_idx = torch.max(prob, dim=1)

                    # Method 1: three thresholds
                    # mask = sigma.le(np.percentile(sigma_cpu, args.thre_u)).float()  # threshold for choosing unlabeled data
                    # mask_p = (sigma.le(np.percentile(sigma_cpu, args.thre_weak))
                    #           & sigma.ge(np.percentile(sigma_cpu,args.thre_strong))).float() # choosing representations to update prototypes

                    # Method 2: same threshold
                    # mask = sigma.le(np.percentile(sigma_cpu, 90)).float()
                    # if iter_num > 20000:
                    #     mask = sigma.le(np.percentile(sigma_cpu, 80)).float()
                    # mask_p = mask

                    # Method 3: cls threshold
                    sigma_channel_last = sigma.permute(0, 2, 3, 1)  # b h w c
                    threshold = [95,97,97,97]
                    mask = torch.zeros_like(max_idx).float().cuda()
                    for idx in range(num_classes):
                        mask_c = max_idx == idx
                        if torch.sum(mask_c) == 0:
                            continue
                        sigma_mean = torch.mean(sigma_channel_last, dim=-1)
                        sigma_mean_c = sigma_mean[mask_c.bool()]
                        sigma_mean_c_cpu = sigma_mean_c.clone().detach().cpu().numpy()
                        mask_p_c = sigma_mean.le(np.percentile(sigma_mean_c_cpu, threshold[idx])).float() * mask_c.float()
                        mask += mask_p_c

                    # visulize ------------------------------------------------------
                    # sigma = sigma[0].mean(0)
                    # sigma = torch.sqrt(sigma)
                    # mu = mu[0].mean(0)
                    # img = volume_batch_ul[0][0]
                    # pred = max_idx[0]
                    # pred_prob = max_probs[0]
                    # refine = mask[0]
                    # gt = label_batch_ul[0]
                    # sigma_np = sigma.detach().cpu().numpy()
                    # mu_np = mu.detach().cpu().numpy()
                    # img_np = img.detach().cpu().numpy()
                    # pred_np = pred.detach().cpu().numpy()
                    # pred_prob_np = pred_prob.detach().cpu().numpy()
                    # refine_np = refine.detach().cpu().numpy()
                    # gt_np = gt.detach().cpu().numpy()
                    # # np.savetxt("a_{}_coarse.txt".format(i), pred_np[:,:,i])
                    # # np.savetxt("a_{}_prob.txt".format(i), pred_prob_np[:,:,i])
                    # np.savetxt("sigma.txt", sigma_np)
                    # # np.savetxt("img.txt", img_np)
                    # # np.savetxt("mu.txt", mu_np)
                    # # np.savetxt("a_{}_refine.txt".format(i), refine_np[:,:,i])
                    # # np.savetxt("gt.txt", gt_np)
                    # assert 1==0
                    # visulize -----------------------------------------------------------

                ## CutMix ###########
                mix_volume_ul, mix_label_ul, mask = util.cut_mix_2d(volume_batch_ul, max_idx, mask)
                volume_batch = torch.cat((volume_batch_l, mix_volume_ul), dim=0)
                max_idx = mix_label_ul

                outputs = s_model(x_2d=volume_batch)
                cls_seg = outputs["cls_seg"]  # b,2,h,w
                proto_seg = outputs["proto_seg"]

                mu = outputs["mu"]
                sigma_sq = outputs["sigma"]

                # log the sigma--------------------------------------------
                sigma = torch.mean(sigma_sq, dim=1)
                sigma_l = sigma[:labeled_bs]
                sigma_cpu = sigma_l.clone().detach().cpu().numpy()
                sigma_cpu_30 = np.percentile(sigma_cpu, 30)
                sigma_cpu_60 = np.percentile(sigma_cpu, 60)
                sigma_cpu_85 = np.percentile(sigma_cpu, 85)
                sigma_cpu_95 = np.percentile(sigma_cpu, 95)
                log_dict['l_sigma_30'] = sigma_cpu_30
                log_dict['l_sigma_60'] = sigma_cpu_60
                log_dict['l_sigma_85'] = sigma_cpu_85
                log_dict['l_sigma_95'] = sigma_cpu_95
                # ----------END--------------------------------------------

                label_batch = torch.cat((label_batch_l, max_idx), dim=0)
                max_idx_conf = max_idx + (1 - mask) * num_classes
                label_batch_conf = torch.cat((label_batch_l, max_idx_conf), dim=0)

                if args.sample == 'reliability':
                    mask_p = cal_sampling_reliability_mask(label=label_batch_conf, sigma_sq=sigma_sq)
                elif args.sample == 'coreset':
                    mask_p = cal_sampling_mask_coreset_fixNum(label=label_batch_conf, mu=mu,
                                                              num_list=[args.coresize for i in range(num_classes)])

                # update the prototypes
                if iter_num % args.proto_update_interval == 0:
                    s_model.prototype_update(mu, sigma_sq, label_batch, mask_p)

                # supervised loss
                loss_cls_ce, loss_seg_dice = mix_loss(cls_seg[:labeled_bs], label_batch[:labeled_bs])
                loss_cls_2d = args.ce_w * loss_cls_ce + args.dice_w * loss_seg_dice

                loss_proto_ce, loss_proto_dice = mix_loss(proto_seg[:labeled_bs], label_batch[:labeled_bs])
                loss_proto_2d = args.ce_w * loss_proto_ce + args.dice_w * loss_proto_dice

                loss_l = loss_cls_2d + args.proto_w * loss_proto_2d

                ## unsupervised loss

                loss_cls_ce, loss_seg_dice = unsup_mix_loss(cls_seg[labeled_bs:], label_batch[labeled_bs:],weight=mask)
                loss_cls_2d = args.ce_w * loss_cls_ce + args.dice_w * loss_seg_dice

                loss_proto_ce, loss_proto_dice = unsup_mix_loss(proto_seg[labeled_bs:], label_batch[labeled_bs:],weight=mask)
                loss_proto_2d = args.ce_w * loss_proto_ce + args.dice_w * loss_proto_dice

                loss_u = loss_cls_2d + args.proto_w * loss_proto_2d

                if args.constant:
                    consistency_weight = args.consistency
                else:
                    consistency_weight = get_current_consistency_weight(
                        (iter_num - args.pretrainIter) // 100)

                loss = loss_l + consistency_weight * loss_u

                log_dict['loss/loss_sup'] = loss_l
                log_dict['loss/loss_ul'] = loss_u

                logging.info(
                    'iteration %d : lr: %f uncer_lr : %f loss : %f loss_sup : %f loss_ul : %f consistency_weight : %f'
                    % (iter_num, base_lr_, uncer_lr_, loss.item(), loss_l.item(), loss_u.item(), consistency_weight))

            log_dict['loss/loss'] = loss
            optimizer.zero_grad()
            optimizer_uncer.zero_grad()
            # with torch.autograd.detect_anomaly():
            loss.backward()
            optimizer.step()
            if iter_num >= args.linearIter:
                optimizer_uncer.step()
            # update t_model
            update_ema_variables(s_model, t_model, args.ema_decay, iter_num)

            if iter_num > args.linearIter and iter_num % 200 == 0:
                s_model.eval()
                metric_list = []
                for i, sampled_batch in enumerate(valloader):
                    metric_i = val_2d.test_single_volume(sampled_batch["image"], sampled_batch["label"], s_model,
                                                         classes=num_classes)  # (nclass-1)*2
                    metric_list += metric_i

                metric_list = np.reshape(metric_list, (-1, 3, 2)).mean(axis=0)
                for class_i in range(num_classes - 1):
                    log_dict['info/val_{}_dice'.format(class_i + 1)] = metric_list[class_i, 0]
                    log_dict['info/val_{}_hd95'.format(class_i + 1)] = metric_list[class_i, 1]
                performance = np.mean(metric_list, axis=0)[0]
                log_dict['info/val_mean_dice'] = performance

                if performance > best_performance:
                    bestIter = iter_num
                    best_performance = performance
                    save_mode_path = os.path.join(snapshot_path,
                                                  'iter_{}_dice_{}.pth'.format(iter_num, round(best_performance, 4)))
                    save_best_path = os.path.join(snapshot_path, '{}_best_model.pth'.format(args.model))
                    save_s_protomu_path = os.path.join(self_snapshot_path, 's_proto_mu.pt'.format(args.model))
                    save_s_protosigma_path = os.path.join(self_snapshot_path, 's_proto_sigma.pt'.format(args.model))
                    save_t_protomu_path = os.path.join(self_snapshot_path, 't_proto_mu.pt'.format(args.model))
                    save_t_protosigma_path = os.path.join(self_snapshot_path, 't_proto_sigma.pt'.format(args.model))
                    torch.save(s_model.state_dict(), save_mode_path)
                    torch.save(s_model.state_dict(), save_best_path)
                    torch.save(s_model.prototypes_mu, save_s_protomu_path)
                    torch.save(s_model.prototypes_sigma, save_s_protosigma_path)
                    torch.save(t_model.prototypes_mu, save_t_protomu_path)
                    torch.save(t_model.prototypes_sigma, save_t_protosigma_path)
                log_dict['info/Best_dice'] = best_performance

                logging.info('iteration %d : mean_dice : %f' % (iter_num, performance))
                s_model.train()

            wandb.log(log_dict)
            torch.cuda.empty_cache()
            if iter_num >= args.max_iterations:
                break

        lr_scheduler.step()
        if epoch >= args.linearIter // len(trainloader) + 1:
            lr_scheduler_uncer.step()
        if iter_num >= args.max_iterations:
            iterator.close()
            break
    logging.info("best iter is: {}".format(bestIter))
    logging.info("exp: {}".format(args.exp))
    wandb.finish()


if __name__ == "__main__":
    if args.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    # -- path to save models
    if args.exp != 'test':
        if args.sample == 'coreset':
            args.exp = args.exp + '_{}_dim{}_{}kpretrain_weight{}_polylr_bs{}_{}_{}_{}_temp{}_uncerlr{}'.format(
                args.coresize,
                args.embed_dim,
                args.pretrainIter // 1000,
                args.consistency,
                args.labeled_bs,
                args.sim_mode,
                args.sigma_mode,
                args.sigma_trans_mode,
                args.temp,
                args.uncer_lr)
        elif args.sample == 'reliability':
            args.exp = args.exp + '_strong{}_weak{}_dim{}_{}kpretrain_weight{}_polylr_bs{}_{}_{}_{}_temp{}_uncerlr{}'.format(
                args.thre_strong,
                args.thre_weak,
                args.embed_dim,
                args.pretrainIter // 1000,
                args.consistency,
                args.labeled_bs,
                args.sim_mode,
                args.sigma_mode,
                args.sigma_trans_mode,
                args.temp,
                args.uncer_lr)

    self_snapshot_path = "../model/ACDC/{}_labeled/{}/{}".format(args.labelnum, args.exp, args.phase)
    for snapshot_path in [self_snapshot_path]:
        if not os.path.exists(snapshot_path):
            os.makedirs(snapshot_path)

    logging.basicConfig(filename=self_snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    train(args, self_snapshot_path)
