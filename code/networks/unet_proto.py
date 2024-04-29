#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import sys
sys.path.insert(0, "../../")
from networks.uncer_head import Uncertainty_head,Uncertainty_head_inverse
from networks.unetmodel import UNet_DS,UNet


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
            momentum=False,

    ):
        super().__init__()
        self.inchannel = inchannel
        self.nclasses = nclasses
        self.sim_mode = sim_mode
        # proto params
        self.temp = temp
        self.momentum = momentum

        uncer_inchannel = 64

        if backbone == 'unetDS':
            self.backbone = UNet_DS(self.inchannel, self.nclasses, out_dim=embed_dim)
        elif backbone == 'unet':
            self.backbone = UNet(self.inchannel, self.nclasses, out_dim=embed_dim)
            uncer_inchannel = 16

        ##### Init Uncertainty Head #####

        # self.uncer_head = Uncertainty_head(in_feat=uncer_inchannel, out_feat=embed_dim, sigma_mode=sigma_mode,
        #                                        sigma_trans_mode=sigma_trans_mode)
        self.uncer_head = Uncertainty_head_inverse(in_feat=uncer_inchannel, out_feat=embed_dim)


        # initialize after several iterations
        if (proto_mu and proto_sigma) is None:
            # self.prototypes_mu = nn.Parameter(torch.zeros(self.nclasses, embed_dim),
            #                                   requires_grad=False)  # # C,dim
            # self.prototypes_sigma = nn.Parameter(torch.zeros(self.nclasses, embed_dim),
            #                                    requires_grad=False)  # # C,dim
            self.prototypes_mu = torch.zeros(self.nclasses, embed_dim).cuda() # # C,dim
            self.prototypes_sigma = torch.zeros(self.nclasses, embed_dim).cuda() # # C,dim
        else:
            self.prototypes_mu = nn.Parameter(proto_mu, requires_grad=False)
            self.prototypes_sigma = nn.Parameter(proto_sigma, requires_grad=False)


    def warm_up(self,
        x_2d
    ):

        assert len(x_2d.shape) == 4
        classifier2d, _, _ = self.backbone(x_2d)
        return classifier2d


    def forward(
            self,
            x_2d,
            label=None,
            mask=None,
    ):
        """

        :param x_2d: size:(B,1,H,W)
        :param label: (B,H,W)
        :param mask: (B,H,W) indicates the result with high confidence, if None, mask equals all ones
        :return:
        """

        classifier2d, mu, feature2d = self.backbone(x_2d) # cls(B,C,H,W),mu (B,dim,H,W)
        return_dict = {}
        return_dict["cls_seg"] = classifier2d

        # sigma_sq = self.uncer_head(feature2d)  # B, dim, H, W

        inverse_sigma_sq = self.uncer_head(feature2d)  # B, dim, H, W
        sigma_sq = 1 / inverse_sigma_sq

        return_dict["sigma"] = sigma_sq # B, dim, H, W
        return_dict["mu"] = mu # B, dim, H, W

        b,dim,h,w = mu.shape
        mu_view = rearrange(mu, "b dim h w-> (b h w) dim")
        sigma_sq_view = rearrange(sigma_sq, "b dim h w-> (b h w) dim")

        # if prototypes_mu and prototypes_sigma are all zeros, initialize them with current probabilistic embeddings
        tmp = torch.zeros_like(self.prototypes_mu)
        if torch.equal(tmp, self.prototypes_mu) and torch.equal(tmp,self.prototypes_sigma):
            print("Initializing the prototypes!!!!!!!!!!!!")
            label_onehot = label_to_onehot(label, num_class=self.nclasses)
            if mask != None:
                mask_ = mask.unsqueeze(1)
            else:
                mask_ = torch.ones((b, 1, h, w)).cuda()
            flag = self.initialize(mu,sigma_sq,label_onehot,mask_)
            if not flag:
                return_dict["proto_seg"] = classifier2d
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
        return_dict["proto_seg"] = proto_prob

        return return_dict


    def prototype_update(self,
                         mu,
                         sigma_sq,
                         label,
                         mask):

        b, h, w = mask.shape
        label_onehot = label_to_onehot(label, num_class=self.nclasses)
        if mask != None:
            mask_ = mask.unsqueeze(1)
        else:
            mask_ = torch.ones((b, 1, h, w)).cuda()

        self.prototype_learning(
            mu,
            sigma_sq,
            label_onehot,
            mask_,
        )

    def euclidean_sim(self,mu_0, mu_1, sigma_0):
        '''
            d_c(i) = sqrt((⃗xi − p⃗c)T Sc (⃗xi − p⃗c))
            Compute the linear Euclidean distances, i.e. dc(i)
            param: mu_0, mu_1 [BxHxW, 1, dim]  [C,dim]
                   sigma_0 [BxHxW, 1, dim]
            '''

        diff = mu_0 - mu_1
        diff_normed = diff / torch.sqrt(sigma_0)
        dist_normed = torch.norm(diff_normed, dim=-1)
        return -dist_normed

    def mutual_likelihood_score(self,mu_0, mu_1, sigma_0, sigma_1):
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

        for i in range(num_segments):  # num_cls
            valid_pixel = valid_pixel_all[:, i]  # B, H, W
            if valid_pixel.sum() == 0:
                print("Initialization fails, class {} is empty....".format(i))
                return False
            # prototype computing
            with torch.no_grad():
                dev = 1 / torch.sum((1 / sigma[valid_pixel.bool()]), dim=0, keepdim=True)  # 1, dim
                mean = torch.sum((dev / sigma[valid_pixel.bool()]) \
                                      * mu[valid_pixel.bool()], dim=0, keepdim=True)  # 1, dim

                protos_mu_curr.append(mean)
                protos_sigma_curr.append(dev)

        protos_mu_curr = torch.cat(protos_mu_curr, dim=0)  # C, dim
        protos_sigma_curr = torch.cat(protos_sigma_curr, dim=0)

        # self.prototypes_mu = nn.Parameter(protos_mu_curr, requires_grad=False)
        # self.prototypes_sigma = nn.Parameter(protos_sigma_curr, requires_grad=False)
        self.prototypes_mu = protos_mu_curr
        self.prototypes_sigma = protos_sigma_curr

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
        valid_pixel_all = label * mask # (B,C,H,W)
        # Permute representation for indexing" [batch, rep_h, rep_w, feat_num]

        mu = mu.permute(0, 2, 3, 1) # B,H,W,dim
        sigma = sigma.permute(0, 2, 3, 1) # B,H,W,dim

        protos_mu_prev = self.prototypes_mu.detach().clone() # # C,dim
        protos_sigma_prev = self.prototypes_sigma.detach().clone() # # C,dim

        protos_mu_curr = []
        protos_sigma_curr = []

        # num_points_min = torch.min(torch.sum(valid_pixel_all, dim=[0, 2, 3])).long().item()
        # print("minimum sample points: {}".format(num_points_min))

        for i in range(num_segments):  # num_cls
            valid_pixel = valid_pixel_all[:, i] # B, H, W
            if valid_pixel.sum() == 0:
                # continue
                # set the sigma and mu of the misses class as torch.inf and 0 respectively
                if self.momentum:
                    dev = protos_sigma_prev[i].unsqueeze(0)
                    mean = protos_mu_prev[i].unsqueeze(0)
                else:
                    dev = torch.full((1, mu.size(-1)), 1e+32).cuda()
                    mean = torch.zeros((1, mu.size(-1))).cuda()

            else:
                # new prototype computing
                with torch.no_grad():
                    dev = 1 / torch.sum((1 / sigma[valid_pixel.bool()]), dim=0, keepdim=True) # 1, dim
                    mean = torch.sum((dev / sigma[valid_pixel.bool()]) \
                                          * mu[valid_pixel.bool()], dim=0, keepdim=True) # 1, dim

            protos_mu_curr.append(mean)
            protos_sigma_curr.append(dev)

        protos_mu_curr = torch.cat(protos_mu_curr, dim=0) # C, dim
        protos_sigma_curr = torch.cat(protos_sigma_curr, dim=0)

        # Prototype updating

        if self.momentum:
            # Method 2: momentum update
            protos_mu_new, protos_sigma_new = momentum_update(protos_mu_prev, protos_mu_curr,
                                                              protos_sigma_prev, protos_sigma_curr, momentum=0.99)

        else:
            # Method 1: (old+new)
            protos_sigma_new = 1 / torch.add(1 / protos_sigma_prev, 1 / protos_sigma_curr)
            protos_mu_new = torch.add((protos_sigma_new / protos_sigma_prev) * protos_mu_prev,
                                     (protos_sigma_new / protos_sigma_curr) * protos_mu_curr)

        # self.prototypes_mu = nn.Parameter(protos_mu_new, requires_grad=False)
        # self.prototypes_sigma = nn.Parameter(protos_sigma_new, requires_grad=False)

        self.prototypes_mu = protos_mu_new
        self.prototypes_sigma = protos_sigma_new

