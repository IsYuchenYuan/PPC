#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from timm.models.layers import trunc_normal_

import sys
from sklearn.cluster import KMeans
sys.path.insert(0, "../../")
from networks.projector import ProjectionV1
import numpy as np

from networks.HUNet import unet_3D
from networks.unetmodel import UNet_DS


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
            # added params
            proj_dim=256,
            projection="v1",
            l2_norm=True,
            proto_mom=0.999,
            sub_proto_size=2,
            proto=None
    ):
        super().__init__()
        self.inchannel = inchannel
        self.nclasses = nclasses

        # proto params
        self.l2_norm = l2_norm
        self.sub_proto_size = sub_proto_size
        self.proto_mom = proto_mom
        self.projection = projection


        # self.backbone = AttU_Net(img_ch=selbackbonef.inchannel,output_ch=self.nclasses)
        # self.backbone3d = denseUnet3d(num_input=1)

        self.backbone = UNet_DS(self.inchannel, self.nclasses)
        self.backbone3d = unet_3D(in_channels=1,n_classes=self.nclasses)
        # self.backbone3d = VNet(n_channels=1, n_classes=self.nclasses)

        self.final = nn.Sequential(
             nn.Conv3d(64, 64, (3, 3, 3), padding=(1, 1, 1)),
             nn.BatchNorm3d(64),
             nn.ReLU(inplace=True),
             nn.Conv3d(64, self.nclasses, (1, 1, 1))
        )
        in_channels = 64
        # contrast define
        if self.projection == "v1":
            self.proj_head = ProjectionV1(in_channels, proj_dim)  # mix
        else:
            raise NotImplementedError

        # initialize after several iterations
        if proto is None:
            if self.sub_proto_size != 1:
                self.prototypes = nn.Parameter(torch.zeros(self.nclasses, self.sub_proto_size, proj_dim),
                                               requires_grad=False)
            else:
                self.prototypes = nn.Parameter(torch.zeros(self.nclasses, proj_dim),
                                           requires_grad=False)
        else:
            self.prototypes = nn.Parameter(proto, requires_grad=False)
        print(self.prototypes.shape)

        self.feat_norm = nn.LayerNorm(proj_dim)
        self.mask_norm = nn.LayerNorm(nclasses)

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
                return 0.0
            feat_centroids = self.kmeans(feat, 1)  # numpy.array (1, 256)
            feat_center_list.append(feat_centroids)
        proto = np.concatenate(feat_center_list, axis=0)  # numpy.array (n_class, 256)
        proto = torch.from_numpy(proto).float()
        proto = proto.cuda()
        self.prototypes = nn.Parameter(proto, requires_grad=False)
        trunc_normal_(self.prototypes, std=0.02)
        return 1.0

    def initialize_sub(self, features, label, subcluster, n_class):
        """

                :param features: (b*h*w,256)
                :param label: (b*h*w, )
                :return:
                """
        label = label.detach().clone().cpu()
        feat_center_list = []
        for i in range(n_class):
            feat = features[label == i]
            if feat.numel() == 0:
               return 0.0
            feat_centroids = self.kmeans(feat, subcluster)  # numpy.array (subcluster, 256)
            feat_center_list.append(feat_centroids[np.newaxis, :])
        proto = np.concatenate(feat_center_list,axis=0)  # numpy.array (n_class, subcluster, 256)
        proto = torch.from_numpy(proto).float()
        proto = proto.cuda()
        self.prototypes = nn.Parameter(proto, requires_grad=False)
        trunc_normal_(self.prototypes, std=0.02)
        return 1.0


    def prototype_updating(
            self,
            out_feat,
            label
    ):
        """
        :param out_feat: [bs*h*w, dim] pixel feature
        :param label: [bs*h*w] segmentation label
        :param feat_proto_sim: [bs*h*w, cls_num]
        """
        # update the prototypes
        protos = self.prototypes.detach().clone()
        for id_c in range(self.nclasses):
            feat_cls =out_feat[label==id_c] # num, dim
            if feat_cls.numel() == 0:
                continue
            f = torch.mean(feat_cls,dim=0)
            new_value = momentum_update(
                old_value=protos[id_c, :],
                new_value=f,
                momentum=self.proto_mom,
                # debug=True if id_c == 1 else False,
                debug=False,
            )  # [p, dim]
            protos[id_c, :] = new_value  # [cls, p, dim]

        self.prototypes = nn.Parameter(l2_normalize(protos), requires_grad=False)

    def prototype_learning(
            self,
            out_feat,
            nearest_proto_distance,
            label,
            feat_proto_sim,
            weighted_sum=False,
    ):
        """
        :param out_feat: [bs*h*w, dim] pixel feature
        :param nearest_proto_distance: [bs, cls_num, h, w]
        :param label: [bs*h*w] segmentation label
        :param feat_proto_sim: [bs*h*w, sub_cluster, cls_num]
        """

        cosine_similarity = feat_proto_sim.reshape(feat_proto_sim.shape[0], -1)
        proto_logits = cosine_similarity
        # proto_target = label.clone().float()  # (n, )
        proto_target = torch.zeros_like(label).float()  # (n, ) n = h*w


        # clustering for each class, on line
        # update the prototypes
        protos = self.prototypes.detach().clone()
        for id_c in range(self.nclasses):

            init_q = feat_proto_sim[
                ..., id_c
            ]  # n, k, cls => n, k
            init_q = init_q[label == id_c, ...]  #
            if init_q.shape[0] == 0:  # no such class
                # print("no class :",id_c)
                continue

            q, indexs = distributed_sinkhorn(init_q)

            # q: (n, 10) one-hot prototype label -> correspond to L in the paper
            # indexes: (n, ) prototype label # torch.unique(indexs) = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


            feat_c = out_feat[label == id_c, ...]  # [n, dim]

            if weighted_sum:
                weight = 1 - init_q  # 1 - (n, p) => (n, p)
                f = weight.transpose(0, 1) @ feat_c  # (p, n) (n, dim) => (p, dim)
            else:
                f = q.transpose(0, 1) @ feat_c  # (p, n) (n, dim) => (p, dim)
                # f is the  ̄ic,k in the paper

            n = torch.sum(q, dim=0)  # (n, p) => p

            if torch.sum(n) > 0:
                f = F.normalize(f, p=2, dim=-1)  # [p, 720]

                new_value = momentum_update(
                    old_value=protos[id_c, n != 0, :],
                    new_value=f[n != 0, :],
                    momentum=self.proto_mom,
                    # debug=True if id_c == 1 else False,
                    debug=False,
                )  # [p, dim]
                protos[id_c, n != 0, :] = new_value  # [cls, p, dim]

            proto_target[label == id_c] = indexs.float() + (
                    self.sub_proto_size * id_c
            )  # (n, ) cls*k classes totally


        self.prototypes = nn.Parameter(l2_normalize(protos), requires_grad=False)


        # if we don't use proto loss, no need to return  proto_logits, proto_target
        return proto_logits, proto_target

    def warm_up(self,
        x_2d,
        x_3d

    ):

        assert len(x_2d.shape) == 4
        B, C, H, W, D = x_3d.size()

        feature2d, classifer2d = self.backbone(x_2d)  # feat(b*d,64,h,w) cls(b*d,2,h,w)
        classifer2To3 = rearrange(
            classifer2d, "(b d) c h w -> b c h w d", b=B
        )
        feature2To3 = rearrange(
            feature2d, "(b d) c h w -> b c h w d", b=B
        )

        # input3d = torch.cat((x_3d, classifer2To3), dim=1)
        feature3d = self.backbone3d(x_3d)
        final = torch.add(feature3d, feature2To3)
        finalout = self.final(final)

        return classifer2d,finalout

    def forward(
            self,
            x_2d,
            x_3d,
            label=None,
            use_prototype=False,

    ):
        """

        :param x: size:(B*D,C,H,W)
        :param x_3d: size:(B,1,H,W,D)
        :param label: (B*D,H,W)
        :param use_prototype: after several pretraining iterations it will be True
        :return:
        """
        B,C,H,W,D = x_3d.size()
        feature2d, classifer2d = self.backbone(x_2d) # feat(b*d,64,h,w) cls(b*d,2,h,w)
        return_dict = {}
        return_dict["cls_seg"] = classifer2d
        # embedding = feature2d
        # embedding = self.proj_head(feature2d)

        feature2To3 = rearrange(
            feature2d, "(b d) c h w -> b c h w d", b=B
        )
        # classifer2To3 = rearrange(
        #     classifer2d, "(b d) c h w -> b c h w d", b=B
        # )
        # feature3d = self.backbone3d(x_3d)
        # input3d = torch.cat((x_3d, classifer2To3), dim=1)
        feature3d = self.backbone3d(x_3d)
        hybridfeature = torch.add(feature3d, feature2To3)
        finalout = self.final(hybridfeature)
        return_dict["cls_seg_3d"] = finalout

        embedding = rearrange(hybridfeature, "b c h w d -> (b d) c h w ")
        # return high level semantic features to refine pseudo labels
        return_dict["feature"] = embedding

        b, dim, h, w = embedding.shape
        out_feat = rearrange(embedding, "b c h w -> (b h w) c")
        out_feat = self.feat_norm(out_feat)  # (n, dim)
        out_feat = l2_normalize(out_feat)  # cosine sim norm

        # initialize the protos
        tmp = torch.zeros_like(self.prototypes)
        '''判断两个tensor是否相等'''
        if use_prototype and label != None:
            label_expand = label.view(-1)
            if torch.equal(tmp, self.prototypes):
                print("initialize the protos")
                out_feat_np = out_feat.detach().clone().cpu()
                if self.sub_proto_size != 1:
                    initialize = self.initialize_sub(out_feat_np,label_expand,self.sub_proto_size,self.nclasses)
                else:
                    initialize = self.initialize(out_feat_np,label_expand,self.nclasses)
                print(initialize)

        self.prototypes.data.copy_(l2_normalize(self.prototypes))
        # cosine sim
        if self.sub_proto_size != 1:
            feat_proto_sim = torch.einsum(
                "nd,kmd->nmk", out_feat, self.prototypes
            )  # [n, dim], [csl, p, dim] -> [n, p, cls]: n=(b h w)
            # choose the nearest sub-cluster for each class at each pixel
            nearest_proto_distance = torch.amax(feat_proto_sim, dim=1)
            nearest_proto_distance = self.mask_norm(nearest_proto_distance)
        else:
            feat_proto_sim = torch.einsum(
                "nd,kd->nk", out_feat, self.prototypes
            )  # [n, dim], [csl, dim] -> [n, cls]: n=(b h w)
            nearest_proto_distance = self.mask_norm(feat_proto_sim)

        nearest_proto_distance = rearrange(
            nearest_proto_distance, "(b h w) k -> b k h w", b=b, h=h
        ) # [n, cls] -> [b, cls, h, w] -> correspond the s in equ(6)
        return_dict["proto_seg"] = nearest_proto_distance

        if use_prototype:
            if self.sub_proto_size != 1:
                contrast_logits, contrast_target = self.prototype_learning(
                    out_feat,
                    nearest_proto_distance,
                    label_expand,
                    feat_proto_sim,
                )
                return_dict["contrast_logits"] = contrast_logits
                return_dict["contrast_target"] = contrast_target
            else:
                self.prototype_updating(out_feat,label_expand)

        return return_dict


if __name__ == "__main__":
    def test_model():

        x = torch.randn((1,1,128,128,32)).cuda()
        label1 = torch.ones((1,128,128,16)).cuda()
        label2 = torch.zeros((1,128,128,16)).cuda()
        label = torch.cat((label1,label2),dim=3)
        model = UNetProto(
            nclasses=2,
            proj_dim=256,
            projection="v1",
            l2_norm=True,
            proto_mom=0.999,
            sub_proto_size=2,
        ).cuda()

        z1 = model(
            x,
            label=label,
            use_prototype=True
        )
        print(z1["cls_seg"].shape)  # b, c, h, w
        print(z1["proto_seg"].shape)  # b, c, h, w
        print(z1["contrast_logits"].shape)  # b, c, h, w
        print(z1["contrast_target"].shape)  # b, c, h, w
        print(model.prototypes.shape)  # cls, proto, dim
    test_model()

