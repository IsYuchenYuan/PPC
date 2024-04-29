#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.
"""
all the prototype related operations are conducted on the 2d slices.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

import sys
sys.path.insert(0, "../../")
from networks.HUNet import unet3D_prob
from networks.VNet import VNet_prob
from networks.uncer_head import Uncertainty_head
from utils.utils import concat_all_gather

def momentum_update(old_mu, new_mu, old_sigma, new_sigma, momentum):
    update_mu = momentum * old_mu + (1 - momentum) * new_mu
    update_sigma = momentum * old_sigma + (1 - momentum) * new_sigma
    return update_mu, update_sigma


def l2_normalize(x):
    return F.normalize(x, p=2, dim=-1)


def label_to_onehot(inputs, num_class):
    '''
    inputs is class label
    return one_hot label
    dim will be increasee
    '''
    batch_size, image_h, image_w = inputs.shape
    inputs = torch.relu(inputs)
    outputs = torch.zeros([batch_size, num_class, image_h, image_w]).to(inputs.device)
    return outputs.scatter_(1, inputs.unsqueeze(1), 1.0)



class UNetProto(nn.Module):
    def __init__(
            self,
            backbone,
            inchannel,
            nclasses,
            proto_mu=None,
            proto_sigma=None,
            embed_dim=256,
            sigma_mode="radius",
            sigma_trans_mode='sigmoid',
            sim_mode='dist',
            temp=100,
            momentum=False

    ):
        super().__init__()
        self.inchannel = inchannel
        self.nclasses = nclasses
        self.temp = temp
        self.sim_mode = sim_mode
        self.momentum = momentum

        ##### Init Uncertainty Head #####
        self.uncer_head = Uncertainty_head(in_feat=16, out_feat=embed_dim, sigma_mode=sigma_mode,
                                           sigma_trans_mode=sigma_trans_mode)

        if backbone=='unet':
            self.backbone3d = unet3D_prob(in_channels=1, n_classes=self.nclasses, outdim=embed_dim)
        elif backbone=='vnet':
            self.backbone3d = VNet_prob(n_channels=1, n_classes=self.nclasses,outdim=embed_dim,
                                        normalization='batchnorm', has_dropout=True)

        # initialize after several iterations
        if (proto_mu and proto_sigma) is None:
            self.prototypes_mu = nn.Parameter(torch.zeros(self.nclasses, embed_dim),
                                              requires_grad=False)  # # C,dim
            self.prototypes_sigma = nn.Parameter(torch.zeros(self.nclasses, embed_dim),
                                                 requires_grad=False)  # # C,dim
        else:
            self.prototypes_mu = nn.Parameter(proto_mu, requires_grad=False)
            self.prototypes_sigma = nn.Parameter(proto_sigma, requires_grad=False)


    def warm_up(self,
        x

    ):

        out_seg = self.backbone3d(x)
        return out_seg

    def forward(
            self,
            x,
            label=None,
            mask=None,

    ):
        """

        :param x_3d: size:(B,C,H,W,D)
        :param label: (B,C,H,W,D)
        :param mask: (B,C,H,W,D) indicates the result with high confidence, if None, mask equals all ones
        :return:
        """
        B, _, _, _, _,= x.size()

        if label != None:
            label2d = rearrange(
                label, " b h w d-> (b d) h w"
            )
        if mask != None:
            mask2d = rearrange(
                mask, " b h w d-> (b d) h w"
            )

        return_dict = {}
        out_seg_3d, mu_3d, feature3d = self.backbone3d(x)
        return_dict["cls_seg_3d"] = out_seg_3d

        mu = rearrange(mu_3d, "b c h w d -> (b d) c h w")
        feature2d = rearrange(feature3d, "b c h w d -> (b d) c h w")

        sigma_sq = self.uncer_head(feature2d)  # B*D, dim, H, W
        sigma_sq = torch.ones_like(mu) * sigma_sq
        return_dict["sigma"] = sigma_sq # B*D, dim, H, W
        return_dict["mu"] = mu  # B*D, dim, H, W

        b,dim,h,w = mu.shape
        mu_view = rearrange(mu, "b dim h w-> (b h w) dim")
        sigma_sq_view = rearrange(sigma_sq, "b dim h w-> (b h w) dim")

        # if prototypes_mu and prototypes_sigma are all zeros, initialize them with current probabilistic embeddings
        tmp = torch.zeros_like(self.prototypes_mu)
        if torch.equal(tmp, self.prototypes_mu) and torch.equal(tmp,self.prototypes_sigma):
            print("Initializing the prototypes!!!!!!!!!!!!")
            label_onehot = label_to_onehot(label2d, num_class=self.nclasses)
            if mask != None:
                mask_ = mask2d.unsqueeze(1)
            else:
                mask_ = torch.ones((b, 1, h, w)).cuda()
            flag = self.initialize(mu,sigma_sq,label_onehot,mask_)
            if not flag:
                return_dict["proto_seg"] = out_seg_3d
                return return_dict

        # cosine sim
        if self.sim_mode=='euclidean':
            proto_sim = self.euclidean_sim(mu_view.unsqueeze(1), # bhw, 1, dim
                                            self.prototypes_mu,# c,dim
                                            sigma_sq_view.unsqueeze(1))
        else:
            proto_sim = self.mutual_likelihood_score(mu_view.unsqueeze(1),  # bhw, 1, dim
                                                     self.prototypes_mu,  # c,dim
                                                     sigma_sq_view.unsqueeze(1),
                                                     self.prototypes_sigma)
        proto_prob = proto_sim / self.temp  # (bhw,c)
        proto_prob = rearrange(
            proto_prob, "(b h w) c -> b c h w", b=b, h=h
        )
        return_dict["proto_seg"] = rearrange(proto_prob,
                                                     " (b d) c h w -> b c h w d ",b=B)

        return return_dict

    def prototype_update(self,
                         mu,
                         sigma_sq,
                         label,
                         mask):

        if label != None:
            label2d = rearrange(
                label, " b h w d-> (b d) h w"
            )
        if mask != None:
            mask2d = rearrange(
                mask, " b h w d-> (b d) h w"
            )

        b, dim, h, w = mu.shape

        label_onehot = label_to_onehot(label2d, num_class=self.nclasses)
        if mask != None:
            mask_ = mask2d.unsqueeze(1)
        else:
            mask_ = torch.ones((b, 1, h, w)).cuda()

        self.prototype_learning(
            mu,
            sigma_sq,
            label_onehot,
            mask_,
        )

    #### MLS ####
    def mutual_likelihood_score(self, mu_0, mu_1, sigma_0, sigma_1):
        '''
        Compute the MLS
        param: mu_0, mu_1 [BxHxW, 1, dim]  [C,dim]
               sigma_0, sigma_1 [BxHxW, 1, dim] [C,dim]
        '''
        mu_0 = F.normalize(mu_0, dim=-1)
        mu_1 = F.normalize(mu_1, dim=-1)
        up = (mu_0 - mu_1) ** 2
        down = sigma_0 + sigma_1
        mls = -0.5 * (up / down + torch.log(down)).mean(-1)

        return mls  # BxHxW, C

    def euclidean_sim(self, mu_0, mu_1, sigma_0):
        '''
            d_c(i) = sqrt((⃗xi − p⃗c)T Sc (⃗xi − p⃗c))
            Compute the linear Euclidean distances, i.e. dc(i)
            param: mu_0, mu_1 [BxHxW, 1, dim]  [C,dim]
                   sigma_0 [BxHxW, 1, dim] -- inverse of sigma_sq
            '''

        # diff = mu_0 - mu_1
        # diff_normed = diff / torch.sqrt(sigma_0)
        diff = (mu_0 - mu_1) ** 2
        diff_normed = diff / sigma_0
        dist_normed = torch.norm(diff_normed, dim=-1)
        return -dist_normed

    def initialize(self,
                   mu,
                   sigma,
                   label,
                   mask):
        """

        :param mu: the mean of the probabilistic representation in the current batch (B,dim,H,W)
        :param sigma: the variance of the probabilistic representation in the current batch (B,dim,H,W)
        :param label: the one-hot label of the batch data (B,C,H,W)
        :param mask: indicates the high prob pixels of the prediction (B,1,H,W)
        :return:
        """

        num_segments = label.shape[1]  # num_cls
        valid_pixel_all = label * mask  # (B,C,H,W)
        # Permute representation for indexing" [batch, rep_h, rep_w, feat_num]

        mu = mu.permute(0, 2, 3, 1)  # B,H,W,dim
        sigma = sigma.permute(0, 2, 3, 1)  # B,H,W,dim

        protos_mu_curr = []
        protos_sigma_curr = []

        # We gather all representations (mu and sigma) cross mutiple GPUs during this progress
        mu_prt = concat_all_gather(mu)  # For protoype computing on all cards (w/o gradients)
        sigma_prt = concat_all_gather(sigma)
        valid_pixel_all_prt = concat_all_gather(valid_pixel_all)  # For protoype computing on all cards

        for i in range(num_segments):  # num_cls
            valid_pixel_prt = valid_pixel_all_prt[:, i]  # B, H, W
            if valid_pixel_prt.sum() == 0:
                print("Initialization fails, class {} is empty....".format(i))
                return False
            # prototype computing
            with torch.no_grad():
                proto_sigma_ = 1 / torch.sum((1 / sigma_prt[valid_pixel_prt.bool()]), dim=0, keepdim=True)  # 1, dim
                proto_mu_ = torch.sum((proto_sigma_ / sigma_prt[valid_pixel_prt.bool()]) \
                                      * mu_prt[valid_pixel_prt.bool()], dim=0, keepdim=True)  # 1, dim

                protos_mu_curr.append(proto_mu_)
                protos_sigma_curr.append(proto_sigma_)

        protos_mu_curr = torch.cat(protos_mu_curr, dim=0)  # C, dim
        protos_sigma_curr = torch.cat(protos_sigma_curr, dim=0)

        self.prototypes_mu = nn.Parameter(protos_mu_curr, requires_grad=False)
        self.prototypes_sigma = nn.Parameter(protos_sigma_curr, requires_grad=False)

        return True

    def prototype_learning(
            self,
            mu_gather,
            sigma_gather,
            label_gather,
            mask_gather,
    ):
        """

        :param mu_gather: the mean of the probabilistic representation in the current batch (B,dim,H,W) gathered from all gpus
        :param label_gather: the one-hot label of the batch data (B,C,H,W) gathered from all gpus
        :param mask_gather: indicates the high prob pixels of the prediction (B,1,H,W) gathered from all gpus
        :param sigma_gather: the variance of the probabilistic representation in the current batch (B,dim,H,W) gathered from all gpus
        :return:
        """

        num_segments = label_gather.shape[1]  # num_cls
        valid_pixel_all_gather = label_gather * mask_gather # (B,C,H,W)
        # Permute representation for indexing" [batch, rep_h, rep_w, feat_num]

        mu_gather = mu_gather.permute(0, 2, 3, 1) # B,H,W,dim
        sigma_gather = sigma_gather.permute(0, 2, 3, 1) # B,H,W,dim

        protos_mu_prev = self.prototypes_mu.detach().clone() # # C,dim
        protos_sigma_prev = self.prototypes_sigma.detach().clone() # # C,dim

        protos_mu_curr = []
        protos_sigma_curr = []

        # We gather all representations (mu and sigma) cross mutiple GPUs during this progress
        # mu_prt = concat_all_gather(mu)  # For protoype computing on all cards (w/o gradients)
        # sigma_prt = concat_all_gather(sigma)
        # valid_pixel_all_prt = concat_all_gather(valid_pixel_all)  # For protoype computing on all cards

        for i in range(num_segments):  # num_cls
            valid_pixel_gather = valid_pixel_all_gather[:, i] # B, H, W
            if valid_pixel_gather.sum() == 0:
                # continue
                # set the sigma and mu of the misses class as torch.inf and 0 respectively
                if self.momentum:
                    proto_sigma_ = protos_sigma_prev[i].unsqueeze(0)
                    proto_mu_ = protos_mu_prev[i].unsqueeze(0)
                else:
                    proto_sigma_ = torch.full((1, mu_gather.size(-1)), 1e+32).cuda()
                    proto_mu_ = torch.zeros((1, mu_gather.size(-1))).cuda()

            else:
                # new prototype computing
                with torch.no_grad():
                    # 1: conditional independence assumption
                    proto_sigma_ = 1 / torch.sum((1 / sigma_gather[valid_pixel_gather.bool()]), dim=0, keepdim=True) # 1, dim
                    proto_mu_ = torch.sum((proto_sigma_ / sigma_gather[valid_pixel_gather.bool()]) \
                                          * mu_gather[valid_pixel_gather.bool()], dim=0, keepdim=True) # 1, dim


            protos_mu_curr.append(proto_mu_)
            protos_sigma_curr.append(proto_sigma_)

        protos_mu_curr = torch.cat(protos_mu_curr, dim=0) # C, dim
        protos_sigma_curr = torch.cat(protos_sigma_curr, dim=0)

        # Prototype updating

        if self.momentum:
            # Method 2: momentum update
            protos_mu_new, protos_sigma_new = momentum_update(protos_mu_prev, protos_mu_curr,
                                                              protos_sigma_prev, protos_sigma_curr, momentum=0.99)

        else:
            # Method 1: (old+new)
            protos_sigma_new = 1 / torch.add(1 / protos_sigma_prev, 1/protos_sigma_curr)
            protos_mu_new = torch.add((protos_sigma_new / protos_sigma_prev) * protos_mu_prev,
                                     (protos_sigma_new / protos_sigma_curr) * protos_mu_curr)

        self.prototypes_mu = nn.Parameter(protos_mu_new, requires_grad=False)
        self.prototypes_sigma = nn.Parameter(protos_sigma_new, requires_grad=False)




