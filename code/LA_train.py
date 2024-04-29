import argparse
import logging
import sys

import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from einops import rearrange
import wandb

from dataloaders.dataset import *
from networks.unet_proto_3d import UNetProto
from utils import losses, ramps, test_3d_patch, util
from scheduler.my_lr_scheduler import PolyLR

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='../data/LA', help='Name of Experiment')
parser.add_argument('--exp', type=str, default="test",
                    help='experiment_name')  # importance_minle-7_uw0.5_cls9597_
parser.add_argument('--proj', type=str, default='Guassian Prototypes -- LA 8 labeled',
                    help='wandb project name')  # LA 8 labeled # LA 4 labeled
parser.add_argument('--phase', type=str, default='unify', help='training phase')
parser.add_argument('--model', type=str, default='vnet', help='model_name')

parser.add_argument('--max_iterations', type=int, default=15000, help='maximum epoch number to train')  # 15000
parser.add_argument("--pretrainIter", type=int, default=0, help="maximum iteration to pretrain")  # 3000
parser.add_argument("--linearIter", type=int, default=0, help="maximum iteration to pretrain") # 200
parser.add_argument('--max_samples', type=int, default=80, help='maximum samples to train')

parser.add_argument('--sim_mode', type=str, default='dist', help='similarity computation method [dist,euclidean]')
parser.add_argument('--embed_dim', type=int, default=32, help='the dimension of the mu')
parser.add_argument('--sigma_mode', type=str, default='diagonal',
                    help='the type of covariance matrix [radius,diagonal]')
parser.add_argument('--sigma_trans_mode', type=str, default='sigmoid',
                    help='the way to transform sigma_raw to sigma')  # softplus, sigmoid, sigmoidLearn
parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
parser.add_argument('--base_lr', type=float, default=1e-2, help='segmentation network learning rate')  # 1e-2 8e-3
parser.add_argument('--uncer_lr', type=float, default=1e-2, help='uncertainty network learning rate')
parser.add_argument('--min_lr', type=float, default=1e-7, help='minmum lr the scheduler to reach')  # le-6
parser.add_argument('--power', type=float, default=0.9, help='the power of lr scheduler')  # le-6
parser.add_argument('--patch_size', type=list, default=(112, 112, 80), help='patch size of network input')
parser.add_argument('--seed', type=int, default=2, help='random seed') # 1337
parser.add_argument('--num_classes', type=int, default=2, help='output channel of network')
# label and unlabel
parser.add_argument('--batch_size', type=int, default=4, help='batch_size per gpu')
parser.add_argument('--labeled_bs', type=int, default=2, help='labeled_batch_size per gpu')  # 6
parser.add_argument('--labelnum', type=int, default=8, help='labeled data')  # 4,8
# costs
parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
parser.add_argument('--ema_decay', type=float, default=0.99, help='ema_decay')
parser.add_argument('--consistency', type=float, default=2, help='consistency')
parser.add_argument('--constant', type=bool, default=False, help='whether use constant ul weight')
parser.add_argument('--consistency_rampup', type=float, default=120.0, help='consistency_rampup')

parser.add_argument("--pretrain", type=bool, default=True, help="whether to initialize prototype")
parser.add_argument("--dice_w", type=float, default=0.5, help="the weight of dice loss")
parser.add_argument("--ce_w", type=float, default=0.5, help="the weight of ce loss")
parser.add_argument("--proto_w", type=float, default=1, help="the weight of proto loss")
parser.add_argument('--proto_update_interval', type=int, default=200, help='the interval iterations for proto updating')
parser.add_argument("--multiDSC", type=bool, default=True, help="whether use multiDSC")
parser.add_argument("--losstype", type=str, default="ce_dice", help="the type of ce and dice loss")
parser.add_argument('--momentum', type=bool, default=False, help='whether use momentum to update protos')

# temperature for softmax in proto matching
parser.add_argument('--temp', type=float, default=0.5, help='temperature for softmax in proto matching') # 0.5 0.2
parser.add_argument('--thre_u1', type=float, default=95, help='threshold for unlabeled loss in the first 1k iterations')
parser.add_argument('--thre_u2', type=float, default=95,
                    help='threshold for unlabeled loss in the remaining iterations')
parser.add_argument('--thre_weak', type=float, default=80, help='weak threshold for sampling the representations')
parser.add_argument('--thre_strong', type=float, default=10,
                    help='strong threshold for sampling the representations')  # in percentile
parser.add_argument('--sample', type=str, default='reliability',
                    help='the type of sampling')  # reliability/coreset/hybrid/equal
parser.add_argument('--coresize', type=int, default=10000, help='the size of coreset')
parser.add_argument('--device', type=str, help='the device')

parser.add_argument('--wandb', type=bool, default=False, help='whether use wandb')

args = parser.parse_args()

num_classes = args.num_classes
patch_size = args.patch_size

os.environ["CUDA_VISIBLE_DEVICES"] = args.device
if not args.wandb:
    # If you don't want your script to sync to the cloud
    os.environ['WANDB_MODE'] = 'dryrun'

if args.labelnum == 4:
    args.temp = 0.2
    args.proj = 'Det -- LA 4 labeled'

# ----------------------------- START of sampling method ------------------------------------- #

# Method 1: Reliability
def cal_sampling_reliability_mask(label, sigma_sq):
    """
    :param sigma_sq: (B*D,C,H,W)
    :param label: (B,H,W,D)
    :return:
    """
    B = label.size(0)
    sigma = rearrange(sigma_sq, "(b d) c h w -> b c h w d ", b=B)
    sigma_channel_last = sigma.permute(0, 2, 3, 4, 1)  # b h w d c
    sample_map = torch.zeros_like(label)
    for idx in range(num_classes):
        sigma_mean = torch.mean(sigma_channel_last, dim=-1)  # b h w d
        mask_p_c = torch.zeros_like(label)
        mask = label == idx
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

# ----------------------------- END of sampling method ------------------------------------- #

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


# ----------------------------- START of loss functions ------------------------------------- #
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

# ----------------------------- START of loss functions ------------------------------------- #

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
    wandb.init(project=args.proj,
               name=args.exp,
               entity='isyuanyc')
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    labeled_bs, unlabeled_bs = args.labeled_bs, args.batch_size - args.labeled_bs

    pretrain = False
    if args.pretrain:
        pretrain = True

    path='../model/LAsingle/8_labeled/ablation/interval_200_strong10_weak80_dim32_3kpretrain_weight2_polylr_bs2_dist_diagonal_sigmoid_temp0.5_uncerlr0.01/' \
                 'unify/'
    checkpoint = path+'vnet_best_model.pth'
    checkpoint_s_mu = path+'s_proto_mu.pt'
    checkpoint_s_sigma = path+'s_proto_sigma.pt'
    checkpoint_t_mu = path+'t_proto_mu.pt'
    checkpoint_t_sigma = path+'t_proto_sigma.pt'

    base_lr = args.base_lr
    uncer_lr = args.uncer_lr
    iter_num = 0
    best_dice = 0.0
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

    db_train = LAHeart(base_dir=args.root_path,
                       split='train',
                       transform=transforms.Compose([
                           RandomRotFlip(),
                           RandomCrop(patch_size),
                           ToTensor(),
                       ]))
    labelnum = args.labelnum
    labeled_idxs = list(range(labelnum))
    unlabeled_idxs = list(range(labelnum, args.max_samples))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, args.batch_size,
                                          args.batch_size - args.labeled_bs)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=4, pin_memory=True,
                             worker_init_fn=worker_init_fn)

    max_epoch = args.max_iterations // len(trainloader) + 1

    optimizer = optim.SGD(s_model.backbone3d.parameters(), lr=base_lr,
                          momentum=0.9, weight_decay=0.0001, nesterov=True)
    optimizer_uncer = optim.SGD(s_model.uncer_head.parameters(), lr=uncer_lr,
                                weight_decay=0.0001,
                                momentum=0.9, nesterov=True)

    lr_scheduler = PolyLR(optimizer, max_epoch, power=args.power, min_lr=args.min_lr)
    lr_scheduler_uncer = PolyLR(optimizer_uncer, max_epoch, power=args.power, min_lr=args.min_lr)

    logging.info("Start self_training")
    logging.info("{} iterations per epoch".format(len(trainloader)))

    s_model.train()
    t_model.train()

    if pretrain:
        start_epoch = iter_num // len(trainloader) + 1
    iterator = tqdm(range(start_epoch, max_epoch), ncols=70)

    for epoch in iterator:
        for _, sampled_batch in enumerate(trainloader):

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
            log_dict['proto_sigma_0'] = torch.mean(s_model.prototypes_sigma[0])
            log_dict['proto_sigma_1'] = torch.mean(s_model.prototypes_sigma[1])

            if iter_num <= args.linearIter:
                cls_seg_3d, _, _ = s_model.warm_up(volume_batch_l)
                loss_cls_ce_3d, loss_seg_dice_3d = mix_loss(cls_seg_3d, label_batch_l)
                loss = args.ce_w * loss_cls_ce_3d + args.dice_w * loss_seg_dice_3d
                logging.info('training linear cls only ... iteration %d : avg loss : %f'
                             % (iter_num, loss.item()))
            elif iter_num <= args.pretrainIter:

                outputs = s_model(x=volume_batch_l,
                                  label=label_batch_l)

                proto_seg = outputs["proto_seg"]  # b,c,h,w,d
                mu = outputs["mu"]  # (b d) c h w
                sigma_sq = outputs["sigma"]  # (b d) c h w

                if args.sample == 'reliability':
                    mask_p = cal_sampling_reliability_mask(label=label_batch_l, sigma_sq=sigma_sq)

                # update the prototypes
                if iter_num % args.proto_update_interval ==0:
                    proportion = s_model.prototype_update(mu, sigma_sq, label_batch_l, mask_p)
                    log_dict['proportion'] = proportion.item()

                loss_proto_ce, loss_proto_dice = mix_loss(proto_seg, label_batch_l)
                loss_proto_3d = args.ce_w * loss_proto_ce + args.dice_w * loss_proto_dice

                cls_seg_3d = outputs["cls_seg_3d"]  # B,c,h,w,d
                loss_cls_ce, loss_seg_dice = mix_loss(cls_seg_3d, label_batch_l)
                loss_cls_3d = args.ce_w * loss_cls_ce + args.dice_w * loss_seg_dice

                loss = args.proto_w * loss_proto_3d + loss_cls_3d

                logging.info('train both cls ... iteration %d : lr: %f uncer_lr : %f avg loss : %f '
                             'loss_proto : %f  loss_cls_3d : %f'
                             % (iter_num, base_lr_, uncer_lr_, loss.item(),
                                loss_proto_3d.item(), loss_cls_3d.item()))

            elif iter_num > args.pretrainIter:

                with torch.no_grad():

                    u_output = t_model(x=volume_batch_ul)
                    u_cls_seg_p = u_output["proto_seg"]
                    prob = F.softmax(u_cls_seg_p, dim=1)
                    sigma = u_output["sigma"]
                    max_probs, max_idx = torch.max(prob, dim=1)

                    # class wise mask
                    sigma_3d = rearrange(sigma, "(b d) c h w -> b c h w d ", b=volume_batch_ul.size(0))
                    sigma_channel_last = sigma_3d.permute(0, 2, 3, 4, 1)  # b h w d c
                    threshold = [95, 97]
                    mask = torch.zeros_like(max_idx).float().cuda()  # b h w d
                    for idx in range(num_classes):
                        mask_c = max_idx == idx
                        if torch.sum(mask_c) == 0:
                            continue
                        sigma_mean = torch.mean(sigma_channel_last, dim=-1)
                        sigma_mean_c = sigma_mean[mask_c.bool()]
                        sigma_mean_c_cpu = sigma_mean_c.clone().detach().cpu().numpy()
                        mask_p_c = sigma_mean.le(
                            np.percentile(sigma_mean_c_cpu, threshold[idx])).float() * mask_c.float()
                        mask += mask_p_c

                ## CutMix ###########
                mix_volume_ul, mix_label_ul,mask = util.cut_mix(volume_batch_ul, max_idx, mask)
                volume_batch = torch.cat((volume_batch_l,mix_volume_ul),dim=0)
                max_idx = mix_label_ul

                outputs = s_model(x=volume_batch)
                mu = outputs["mu"]
                sigma_sq = outputs["sigma"]

                label_batch = torch.cat((label_batch_l, max_idx), dim=0)
                max_idx_conf = max_idx + (1 - mask) * num_classes
                label_batch_conf = torch.cat((label_batch_l, max_idx_conf), dim=0)

                if args.sample == 'reliability':
                    mask_p = cal_sampling_reliability_mask(label=label_batch_conf, sigma_sq=sigma_sq)
                # update the prototypes
                if iter_num % args.proto_update_interval == 0:
                    proportion=s_model.prototype_update(mu, sigma_sq, label_batch, mask_p)
                    log_dict['proportion'] = proportion.item()

                proto_seg = outputs["proto_seg"]
                cls_seg_3d = outputs["cls_seg_3d"]
                # supervised loss
                loss_proto_ce, loss_proto_dice = mix_loss(proto_seg[:labeled_bs], label_batch_l)
                loss_proto_3d = args.ce_w * loss_proto_ce + args.dice_w * loss_proto_dice
                loss_cls_ce, loss_seg_dice = mix_loss(cls_seg_3d[:labeled_bs], label_batch_l)
                loss_cls_3d = args.ce_w * loss_cls_ce + args.dice_w * loss_seg_dice
                loss_l = args.proto_w * loss_proto_3d + loss_cls_3d

                ## unsupervised loss
                loss_proto_ce, loss_proto_dice = unsup_mix_loss(proto_seg[labeled_bs:], max_idx, weight=mask)
                loss_proto_3d = args.ce_w * loss_proto_ce + args.dice_w * loss_proto_dice
                loss_cls_ce, loss_seg_dice = unsup_mix_loss(cls_seg_3d[labeled_bs:], max_idx, weight=mask)
                loss_cls_3d = args.ce_w * loss_cls_ce + args.dice_w * loss_seg_dice
                loss_u = args.proto_w * loss_proto_3d + loss_cls_3d

                if args.constant:
                    consistency_weight = args.consistency
                else:
                    consistency_weight = get_current_consistency_weight(
                        (iter_num - args.pretrainIter) // 100)

                loss = loss_l + consistency_weight * loss_u

                log_dict['loss/loss_sup'] = loss_l.item()
                log_dict['loss/loss_ul'] = loss_u.item()
                log_dict['loss/loss_ul_proto'] = loss_proto_3d.item()
                log_dict['loss/loss_ul_cls'] = loss_cls_3d.item()

                logging.info(
                    'iteration %d : lr: %f uncer_lr : %f loss : %f loss_sup : %f loss_ul : %f consistency_weight : %f'
                    % (iter_num, base_lr_, uncer_lr_, loss.item(), loss_l.item(), loss_u.item(), consistency_weight))

            log_dict['loss/loss'] = loss.item()
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
                dice_sample = test_3d_patch.var_all_case_LA(s_model, num_classes=num_classes, patch_size=patch_size,
                                                            stride_xy=18, stride_z=4)
                if dice_sample > best_dice:
                    bestIter = iter_num
                    best_dice = round(dice_sample, 4)
                    save_mode_path = os.path.join(self_snapshot_path, 'iter_{}_dice_{}.pth'.format(iter_num, best_dice))
                    save_best_path = os.path.join(self_snapshot_path, '{}_best_model.pth'.format(args.model))
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
                    logging.info("save best model to {}".format(save_mode_path))
                log_dict['Var_dice/Dice'] = dice_sample
                log_dict['Var_dice/Best_dice'] = best_dice
                logging.info('iteration %d : mean_dice : %f' % (iter_num, dice_sample))
                s_model.train()

            if iter_num % 500 == 0:
                save_mode_path = os.path.join(self_snapshot_path, 'iter_{}.pth'.format(iter_num))
                torch.save(s_model.state_dict(), save_mode_path)

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
    logging.info("best dice is: {}".format(best_dice))
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

    self_snapshot_path = "../model/LAsingle/{}_labeled/{}/{}".format(args.labelnum, args.exp, args.phase)
    for snapshot_path in [self_snapshot_path]:
        if not os.path.exists(snapshot_path):
            os.makedirs(snapshot_path)

    logging.basicConfig(filename=self_snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    train(args, self_snapshot_path)
