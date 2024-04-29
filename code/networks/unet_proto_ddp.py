#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import sys

sys.path.insert(0, "../../")
from networks.uncer_head import Uncertainty_head
from networks.unetmodel import UNet_DS,UNet
from utils.utils import concat_all_gather

def momentum_update(old_mu, new_mu, old_sigma, new_sigma, momentum):
    update_mu = momentum * old_mu + (1 - momentum) * new_mu
    update_sigma = momentum * old_sigma + (1 - momentum) * new_sigma
    return update_mu, update_sigma


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


#### MLS ####
def mutual_likelihood_score(mu_0, mu_1, sigma_0, sigma_1):
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


class UNetProto(nn.Module):
    def __init__(
            self,
            inchannel,
            nclasses,
            proj_dim=256,
            proto_mu=None,
            proto_sigma=None,
            temp=100,

    ):
        super().__init__()
        self.inchannel = inchannel
        self.nclasses = nclasses

        # proto params
        self.temp = temp
        self.backbone = UNet_DS(self.inchannel, self.nclasses, out_dim=proj_dim)
        # self.backbone = UNet(self.inchannel, self.nclasses, out_dim=proj_dim)
        ##### Init Uncertainty Head #####
        self.uncer_head = Uncertainty_head(in_feat=64, out_feat=proj_dim)

        # initialize after several iterations
        if (proto_mu and proto_sigma) is None:
            self.prototypes_mu = nn.Parameter(torch.zeros(self.nclasses, proj_dim),
                                              requires_grad=False)  # # C,dim
            self.prototypes_sigma = nn.Parameter(torch.zeros(self.nclasses, proj_dim),
                                                 requires_grad=False)  # # C,dim


        else:
            self.prototypes_mu = nn.Parameter(proto_mu, requires_grad=False)
            self.prototypes_sigma = nn.Parameter(proto_sigma, requires_grad=False)

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
            valid_pixel = valid_pixel_all_prt[:, i]  # B, H, W
            if valid_pixel.sum() == 0:
                print("Initialization fails, class {} is empty....".format(i))
                return False
            # new prototype computing
            with torch.no_grad():
                proto_sigma_ = 1 / torch.sum((1 / sigma_prt[valid_pixel.bool()]), dim=0, keepdim=True)  # 1, dim
                proto_mu_ = torch.sum((proto_sigma_ / sigma_prt[valid_pixel.bool()]) \
                                      * mu_prt[valid_pixel.bool()], dim=0, keepdim=True)  # 1, dim

                protos_mu_curr.append(proto_mu_)
                protos_sigma_curr.append(proto_sigma_)

        protos_mu_curr = torch.cat(protos_mu_curr, dim=0)  # C, dim
        protos_sigma_curr = torch.cat(protos_sigma_curr, dim=0)

        self.prototypes_mu = nn.Parameter(protos_mu_curr, requires_grad=False)
        self.prototypes_sigma = nn.Parameter(protos_sigma_curr, requires_grad=False)

        return True

    def prototype_learning(
            self,
            mu,
            sigma,
            label,
            mask,
    ):
        """

        :param mu: the mean of the probabilistic representation in the current batch (B,dim,H,W)
        :param label: the one-hot label of the batch data (B,C,H,W)
        :param mask: indicates the high prob pixels of the prediction (B,1,H,W)
        :param sigma: the variance of the probabilistic representation in the current batch (B,dim,H,W)
        :param prob: the prediction (B,C,H,W)
        :return:
        """

        num_segments = label.shape[1]  # num_cls
        valid_pixel_all = label * mask  # (B,C,H,W)
        # Permute representation for indexing" [batch, rep_h, rep_w, feat_num]

        mu = mu.permute(0, 2, 3, 1)  # B,H,W,dim
        sigma = sigma.permute(0, 2, 3, 1)  # B,H,W,dim

        # We gather all representations (mu and sigma) cross mutiple GPUs during this progress
        mu_prt = concat_all_gather(mu)  # For protoype computing on all cards (w/o gradients)
        sigma_prt = concat_all_gather(sigma)
        valid_pixel_all_prt = concat_all_gather(valid_pixel_all)  # For protoype computing on all cards

        protos_mu_prev = self.prototypes_mu.detach().clone()  # # C,dim
        protos_sigma_prev = self.prototypes_sigma.detach().clone()  # # C,dim

        protos_mu_curr = []
        protos_sigma_curr = []

        for i in range(num_segments):  # num_cls
            valid_pixel = valid_pixel_all_prt[:, i]  # B, H, W
            if valid_pixel.sum() == 0:
                # continue
                # set the sigma and mu of the misses class as torch.inf and 0 respectively
                proto_sigma_ = torch.full((1, mu_prt.size(-1)), torch.inf).cuda()
                proto_mu_ = torch.zeros((1, sigma_prt.size(-1))).cuda()
            else:
                # new prototype computing
                with torch.no_grad():
                    # 1: conditional independence assumption
                    proto_sigma_ = 1 / torch.sum((1 / sigma_prt[valid_pixel.bool()]),
                                                 dim=0, keepdim=True)  # 1, dim
                    proto_mu_ = torch.sum((proto_sigma_ / sigma_prt[valid_pixel.bool()]) \
                                          * mu_prt[valid_pixel.bool()], dim=0, keepdim=True)  # 1, dim

            protos_mu_curr.append(proto_mu_)
            protos_sigma_curr.append(proto_sigma_)

        protos_mu_curr = torch.cat(protos_mu_curr, dim=0)  # C, dim
        protos_sigma_curr = torch.cat(protos_sigma_curr, dim=0)

        # Prototype updating
        # Method 1: (old+new)
        protos_sigma_new = 1 / torch.add(1 / protos_sigma_prev, 1 / protos_sigma_curr)
        protos_mu_new = torch.add((protos_sigma_new / protos_sigma_prev) * protos_mu_prev,
                                  (protos_sigma_new / protos_sigma_curr) * protos_mu_curr)

        # Method 2: momentum update
        # protos_mu_new,protos_sigma_new = momentum_update(protos_mu_prev,protos_mu_curr,
        #                                                  protos_sigma_prev,protos_sigma_curr,momentum=0.99)

        self.prototypes_mu = nn.Parameter(protos_mu_new, requires_grad=False)
        self.prototypes_sigma = nn.Parameter(protos_sigma_new, requires_grad=False)

    def warm_up(self,
                x_2d
                ):

        assert len(x_2d.shape) == 4
        classifer2d, _, _ = self.backbone(x_2d)
        return classifer2d

    def forward(
            self,
            x_2d,
            label=None,
            mask=None,
            update_prototype=False,

    ):
        """

        :param x_2d: size:(B,1,H,W)
        :param label: (B,H,W)
        :param mask: (B,H,W) indicates the result with high confidence, if None, mask equals all ones
        :param update_prototype: whether update the prototype
        :return:
        """

        classifer2d, _mu, feature2d = self.backbone(x_2d)  # cls(B,C,H,W),mu (B,dim,H,W),feat(B,f_dim,H,W)
        _, num_cls, _, _ = classifer2d.shape
        return_dict = {}
        return_dict["cls_seg"] = classifer2d

        _sigma = self.uncer_head(feature2d)  # B, dim, H, W
        return_dict["sigma"] = _sigma
        b, dim, h, w = _mu.shape
        mu = rearrange(_mu, "b dim h w-> (b h w) dim")
        sigma = rearrange(_sigma, "b dim h w-> (b h w) dim")

        # if prototypes_mu and prototypes_sigma are all zeros, initialize them with current probabilistic embeddings
        tmp = torch.zeros_like(self.prototypes_mu)
        if torch.equal(tmp, self.prototypes_mu) and torch.equal(tmp, self.prototypes_sigma):
            print("Initializing the prototypes!!!!!!!!!!!!")
            label_onehot = label_to_onehot(label, num_class=num_cls)
            if mask != None:
                mask_ = mask.unsqueeze(1)
            else:
                mask_ = torch.ones((b, 1, h, w)).cuda()
            flag = self.initialize(_mu, _sigma, label_onehot, mask_)
            if not flag:
                return_dict["proto_seg"] = classifer2d
                return return_dict

        # cosine sim
        proto_sim = mutual_likelihood_score(mu.unsqueeze(1),  # bhw, 1, dim
                                            self.prototypes_mu,  # c,dim
                                            sigma.unsqueeze(1),
                                            self.prototypes_sigma)
        proto_prob = proto_sim / self.temp  # (bhw,c)
        proto_prob = rearrange(
            proto_prob, "(b h w) c -> b c h w", b=b, h=h
        )
        return_dict["proto_seg"] = proto_prob

        if update_prototype:

            label_onehot = label_to_onehot(label, num_class=num_cls)
            if mask != None:
                mask_ = mask.unsqueeze(1)
            else:
                mask_ = torch.ones((b, 1, h, w)).cuda()

            self.prototype_learning(
                _mu,
                _sigma,
                label_onehot,
                mask_,

            )

        return return_dict


