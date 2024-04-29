#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.layers import trunc_normal_

import sys
from sklearn.cluster import KMeans
sys.path.insert(0, "../../")
import numpy as np
from networks.VNet import VNet_prob

def momentum_update(old_value, new_value, momentum, debug=False):
    update = momentum * old_value + (1 - momentum) * new_value
    if debug:
        print(
            "# old prot: {:.3f} x |{:.3f}|, new val: {:.3f} x |{:.3f}|, result= |{:.3f}|".format(
                momentum,
                torch.norm(old_value, p=2),
                (1 - momentum),
                torch.norm(new_value, p=2),
                torch.norm(update, p=2),
            )
        )
    return update


def l2_normalize(x):
    return F.normalize(x, p=2, dim=-1)


class UNetProto(nn.Module):
    def __init__(
            self,
            inchannel,
            nclasses,
            embed_dim=256,
            l2_norm=True,
            proto_mom=0.999,
            proto=None
    ):
        super().__init__()
        self.inchannel = inchannel
        self.nclasses = nclasses

        # proto params
        self.l2_norm = l2_norm
        self.proto_mom = proto_mom

        self.backbone3d = VNet_prob(n_channels=1, n_classes=self.nclasses, outdim=embed_dim,
                                    normalization='batchnorm', has_dropout=True)

        # initialize after several iterations
        if proto is None:
            self.prototypes = torch.zeros(self.nclasses, embed_dim).cuda()
        else:
            self.prototypes = proto
        print(self.prototypes.shape)

        self.feat_norm = nn.LayerNorm(embed_dim)
        self.mask_norm = nn.LayerNorm(nclasses)

    def warm_up(self,
        x

    ):

        out_seg = self.backbone3d(x)
        return out_seg

    def forward(
            self,
            x,
            label=None,

    ):
        """
        :param x: size:(B,C,H,W,D)
        :param label: (B,H,W,D)
        :return:
        """
        B = x.size(0)
        if label != None:
            label2d = rearrange(
                label, " b h w d-> (b d) h w"
            )
        return_dict = {}
        out_seg_3d, feature3d, _ = self.backbone3d(x)
        return_dict["cls_seg_3d"] = out_seg_3d

        embedding = rearrange(feature3d, "b c h w d -> (b d) c h w ")
        # return high level semantic features to refine pseudo labels

        b, dim, h, w = embedding.shape
        out_feat = rearrange(embedding, "B c h w -> (B h w) c")
        out_feat = self.feat_norm(out_feat)  # (n, dim)
        out_feat = l2_normalize(out_feat)  # cosine sim norm
        return_dict["feature"] = out_feat

        # initialize the protos
        tmp = torch.zeros_like(self.prototypes)
        '''判断两个tensor是否相等'''
        # if prototypes_mu and prototypes_sigma are all zeros, initialize them with current probabilistic embeddings
        if torch.equal(tmp, self.prototypes):
            label_expand = label2d.view(-1)
            print("initialize the protos")
            out_feat_np = out_feat.detach().clone().cpu()
            flag = self.initialize(out_feat_np,label_expand,self.nclasses)
            if not flag:
                return_dict["proto_seg"] = out_seg_3d
                return return_dict

        self.prototypes = l2_normalize(self.prototypes)
        # cosine sim

        feat_proto_sim = torch.einsum(
            "nd,kd->nk", out_feat, self.prototypes
        )  # [n, dim], [csl, dim] -> [n, cls]: n=(b h w)
        nearest_proto_distance = self.mask_norm(feat_proto_sim)

        nearest_proto_distance = rearrange(
            nearest_proto_distance, "(b h w) k -> b k h w", b=b, h=h
        ) # [n, cls] -> [b, cls, h, w] -> correspond the s in equ(6)
        return_dict["proto_seg"] = rearrange(nearest_proto_distance,
                                                     " (b d) c h w -> b c h w d ",b=B)

        return return_dict

    def prototype_update(self,
                         out_feat,
                         label,
                         mask=None):
        """

        :param out_feat: (n,dim)
        :param label: (B,H,W,D)
        :param mask: (B,H,W,D)
        :return:
        """
        if label != None:
            label2d = rearrange(
                label, " b h w d-> (b d) h w"
            )

        if mask == None:
            mask = torch.ones_like(label).cuda()
        mask2d = rearrange(
            mask, " b h w d-> (b d) h w"
        )
        label_expand = label2d.view(-1)
        mask_expand = mask2d.view(-1)

        self.prototype_learning(out_feat, label_expand, mask_expand)

    def kmeans(self, feature, sub_proto_size):
        """

        :param feature: size:(n,256) n is the number of features whose label is 1 or 0
        :param sub_proto_size:
        :return: cluster center for each clustern size:(sub_proto_size,256)
        """
        kmeans = KMeans(n_clusters=sub_proto_size, random_state=0).fit(feature)
        centroids = kmeans.cluster_centers_
        return centroids

    def initialize(self, features, label, n_class):
        label = label.detach().clone().cpu()
        feat_center_list = []
        for i in range(n_class):
            feat = features[label == i]
            if feat.numel() == 0:
                print("Initialization fails, class {} is empty....".format(i))
                return False
            feat_centroids = self.kmeans(feat, 1)  # numpy.array (1, 256)
            feat_center_list.append(feat_centroids)
        proto = np.concatenate(feat_center_list, axis=0)  # numpy.array (n_class, 256)
        proto = torch.from_numpy(proto).float()
        self.prototypes = proto.cuda()
        trunc_normal_(self.prototypes, std=0.02)
        return True


    def prototype_learning(
            self,
            out_feat,
            label,
            mask
    ):
        """
        :param out_feat: [bs*h*w, dim] pixel feature
        :param label: [bs*h*w] segmentation label
        :param mask: [bs*h*w] binary mask which sets zero to unwanted pixels
        """
        # update the prototypes
        label = label + (1 - mask) * self.nclasses # filter out unwanted pixels
        protos = self.prototypes.detach().clone()
        with torch.no_grad():
            for id_c in range(self.nclasses):
                feat_cls =out_feat[label==id_c] # num, dim
                if feat_cls.numel() == 0:
                    continue
                f = torch.mean(feat_cls,dim=0)
                new_value = momentum_update(
                    old_value= protos[id_c, :],
                    new_value=f,
                    momentum=self.proto_mom,
                    # debug=True if id_c == 1 else False,
                    debug=False,
                )  # [p, dim]
                protos[id_c, :] = new_value
        self.prototypes = protos # [cls, p, dim]


