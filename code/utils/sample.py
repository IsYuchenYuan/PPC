"""
sampling method for pixel selection
"""

from einops import rearrange
import torch
import numpy as np
import torch.nn.functional as F
from sklearn.cluster import KMeans

# Method 1: sample the least uncertainty pixels
def cal_smapling_rate(label,sigma_sq,num_classes,thre_weak,thre_strong):
    """

    :param sigma_sq: (B*D,C,H,W)
    :param label: (B,H,W,D)
    :return:
    """
    sigma = rearrange(sigma_sq, "(b d) c h w -> b c h w d ", b=label.size(0))
    sigma_channel_last = sigma.permute(0, 2, 3, 4, 1) # b h w d c
    sampel_nums = []
    for idx in range(num_classes):
        sigma_c = sigma_channel_last[label == idx]
        sigma_c = torch.mean(sigma_c, dim=-1)
        sigma_c_cpu = sigma_c.clone().detach().cpu().numpy()
        mask_p_c = (sigma_c.le(np.percentile(sigma_c_cpu, thre_weak))
                    & sigma_c.ge(np.percentile(sigma_c_cpu,
                                                  thre_strong))).float()  # choosing representations to update prototypes

        sampel_nums.append(torch.sum(mask_p_c))

    print("sampel_nums {}".format(sampel_nums))


    sample_rate = [min(sampel_nums) / num for num in sampel_nums]
    return sample_rate


def cal_sampling_mask(label, sam_rate, num_classes):
    sample_map = torch.zeros_like(label).float()
    vol_shape = label.shape
    for idx in range(num_classes):
        prob = 1 - sam_rate[idx]
        rand_map = torch.rand(vol_shape).cuda() * (label == idx)
        rand_map = (rand_map > prob) * 1.0
        sample_map += rand_map
    return sample_map

# Method 2: Sample the coreset pixels
def cal_smapling_rate_coreset(label,mu,num_classes):
    """

        :param mu: (B*D,C,H,W)
        :param label: (B,H,W,D)
        :return:
        """
    mu = rearrange(mu, "(b d) c h w -> b c h w d ", b=label.size(0))
    mu_dimMean = torch.mean(mu,dim=1) # b h w d
    sample_rate_list = []
    for idx in range(num_classes):
        mask = label==idx
        mu_c_mean = torch.mean(mu_dimMean[mask.bool()])
        dist_sum = torch.sum((mu_dimMean[mask.bool()]-mu_c_mean)**2)
        dist_all = (mu_dimMean - mu_c_mean) ** 2
        dist_c = dist_all * mask
        qx_c = 0.5*1/torch.sum(label==idx) + 0.5*dist_c/dist_sum
        sample_rate_list.append(qx_c)
    return sample_rate_list


def cal_smapling_rate_coreset_cosine(label,mu,num_classes):
    """

        :param mu: (B*D,C,H,W)
        :param label: (B,H,W,D)
        :return:
        """
    mu = rearrange(mu, "(b d) c h w -> b c h w d ", b=label.size(0))
    mu_channel_last = mu.permute(0, 2, 3, 4, 1)  # b h w d c
    sample_rate_list = []
    for idx in range(num_classes):
        mask = label==idx
        mu_c_mean = torch.mean(mu_channel_last[mask.bool()],dim=0).unsqueeze(0) # 1,c
        dist_sum = torch.sum(1- F.cosine_similarity(mu_channel_last[mask.bool()],mu_c_mean,dim=-1))
        dist_all = 1 - F.cosine_similarity(mu_channel_last,mu_c_mean,dim=-1)
        dist_c = dist_all * mask
        qx_c = 0.5*1/torch.sum(label==idx) + 0.5*dist_c/dist_sum
        sample_rate_list.append(qx_c)
    return sample_rate_list


def cal_sampling_mask_coreset(sample_rate_list):
    sample_rate = torch.zeros_like(sample_rate_list[0]).float().cuda()
    for sample_rate_c in sample_rate_list:
        sample_rate += sample_rate_c
    rand_map = torch.rand(sample_rate.size()).cuda()
    sample_map = (rand_map > sample_rate) * 1.0
    return sample_map


def cal_sampling_mask_coreset_fixNum(sample_rate_list,num):
    sample_map = torch.zeros_like(sample_rate_list[0]).float().cuda()
    for sample_rate_c in sample_rate_list:
        b,h,w,d = sample_rate_c.size()
        sample_map_c = torch.zeros_like(sample_rate_c).float().cuda()
        p = rearrange(sample_rate_c, "b h w d ->  (b h w d)").detach().cpu().numpy()
        p = p / np.linalg.norm(p, ord=1)
        sample_map_flatten = rearrange(sample_map_c, "b h w d ->  (b h w d)")
        sample_map_flatten_cpu = sample_map_flatten.detach().cpu().numpy()
        sample_idx = np.random.choice(np.arange(sample_map_flatten_cpu.shape[0]), num, p=p)
        sample_map_flatten_cpu[sample_idx] = 1
        sample_map_c = torch.from_numpy(sample_map_flatten_cpu).cuda()
        sample_map_c = rearrange(sample_map_c, "(b h w d) -> b h w d ",b=b,h=h,w=w)
        sample_map += sample_map_c
    return sample_map


# Method 3: NMS
def NMS(feats, mask, budget, threshold):
    """

    :param feats: (B,H,W,D,C) mu of the pixels
    :param mask: (B,H,W,D) class mask
    :param budget: int, the number of sampling pixels
    :param threshold: float, the similarity threshold
    :return:
    """
    b, h, w, d = mask.size()
    # step 1: calculate the cluster the features
    cluster = torch.mean(feats[mask.bool()],dim=0).unsqueeze(0) # 1,C
    # step 2: calculate the cosine sim between cluster and pixels of the same class
    sim_c = F.cosine_similarity(feats,cluster,dim=-1) # B,H,W,D
    sim_c[~mask.bool()] = 2 # ensure the other classes can not be retrieved, set the similarity very high
    sim_c_flatten = rearrange(sim_c, "b h w d ->  (b h w d)")
    # step 3: filter out the pixels that share a high degree of similarity with some other and are close to the cluster center.
    sample_map_c = torch.zeros_like(sim_c_flatten)
    sim_order,sim_order_idx = torch.sort(sim_c_flatten)

    if torch.sum(mask) <= budget:
        return mask

    indx=[]
    num=0
    sample_idx=0

    while num < budget:
        indx.append(sample_idx)
        num += 1
        nextidx = (torch.nonzero(sim_order[sim_order!=2] >= (sim_order[sample_idx]+threshold)).T)[0]
        if nextidx.size(-1) == 0:
            print("not enough for 500")
            break
        sample_idx = nextidx[0]

    # print("final index: {}".format(indx[-1]))

    select_idx = sim_order_idx[torch.tensor(indx).cuda()]
    if len(select_idx)!=0:
        sample_map_c[select_idx] = 1
    sample_map_c = rearrange(sample_map_c, "(b h w d) -> b h w d ", b=b, h=h, w=w)
    return sample_map_c


def cal_sampling_mask_NMS(label,mu,num_classes,budget,threshold):
    """
    :param mu: (B*D,C,H,W)
    :param label: (B,H,W,D)
    :param num_classes: the number of class
    :param budget: the number of sampling pixels
    :param threshold: float, the similarity threshold
    :return:
    """
    sample_map = torch.zeros_like(label).float().cuda()
    mu = rearrange(mu, "(b d) c h w -> b c h w d ", b=label.size(0))
    mu_channel_last = mu.permute(0, 2, 3, 4, 1)  # b h w d c
    for idx in range(num_classes):
        mask = label == idx
        sample_map_c = NMS(mu_channel_last,mask,budget,threshold)
        sample_map += sample_map_c
    return sample_map



def cluster(feats, sigma, mask, sub_clusters):
    """

    :param feats: (B,H,W,D,C) mu of the pixels
    :param sigma: (B,H,W,D,C) mu of the pixels
    :param mask: (B,H,W,D) class mask
    :param sub_clusters: int, the number of pixels selected for updating the prototype
    :return:
    """
    feats_c = feats[mask.bool()] # (N, C) N is the number of pixels belonging to cth class
    sigma_c = sigma[mask.bool()]

    keep = {} # store the mu and sigma of the selected pixels

    num = feats_c.size(0)
    if num <= sub_clusters:
        keep['mu'] = feats_c
        keep['sigma'] = sigma_c
        return keep
    # step 1: calculate the cluster the features
    feats_c_np = feats_c.detach().cpu().numpy()
    kmeans = KMeans(n_clusters=sub_clusters, random_state=0).fit(feats_c_np)
    cluster_lbs = kmeans.labels_

    for i in range(sub_clusters):
        cluster_c = feats_c[cluster_lbs==i]
        cluster_sigma_c =  sigma_c[cluster_lbs==i]
        sort_idx = torch.argsort(cluster_sigma_c,dim=-1)
        keep['mu']=(cluster_c[sort_idx[0]])
        keep['sigma']=(cluster_sigma_c[sort_idx[0]])
    return keep


def cal_sampling_mask_cluster(label,mu,sigma,sub_clusters,num_classes):
    """
    :param mu: (B*D,C,H,W)
    :param sigma: (B*D,C,H,W)
    :param label: (B,H,W,D)
    :param sub_clusters: int, the number of pixels selected for updating the prototype
    :return:
    """
    mu = rearrange(mu, "(b d) c h w -> b c h w d ", b=label.size(0))
    sigma = rearrange(sigma, "(b d) c h w -> b c h w d ", b=label.size(0))
    mu_channel_last = mu.permute(0, 2, 3, 4, 1)  # b h w d c
    sigma_channel_last = sigma.permute(0, 2, 3, 4, 1)  # b h w d c

    updatelist=[]
    for idx in range(num_classes):
        mask = label == idx
        sample_dict = cluster(mu_channel_last,sigma_channel_last,mask,sub_clusters)
        updatelist.append(sample_dict)
    return updatelist


def cal_sampling_mask_uncer_fixNum(label,sigma,num_points,importance_sample_ratio):
    """
    :param label: (B,H,W,D)
    :param sigma: (B*D,C,H,W)
    :param num_points: the number of coreset. INT
    :param importance_sample_ratio (float): Ratio of points that are sampled via importnace sampling.
    :return:
    """

    sigma = rearrange(sigma, "(b d) c h w -> b c h w d ", b=label.size(0))
    sigma_mean = torch.mean(sigma,dim=1) # b h w d
    b, h, w, d = label.size()
    sample_map = torch.zeros_like(label).float().cuda()

    num_uncertain_points = int(importance_sample_ratio * num_points)
    num_random_points = num_points - num_uncertain_points

    for idx in range(num_classes):
        sample_map_c = torch.zeros_like(label).float().cuda()
        mask = label == idx
        # importance sample
        sigma_mean_c = sigma_mean * mask
        sigma_c = rearrange(sigma_mean_c, "b h w d ->  (b h w d)")
        mask_flatten = rearrange(mask, "b h w d ->  (b h w d)")
        uncer_idx = torch.topk(sigma_c, k=num_uncertain_points)[1]
        sample_map_flatten = rearrange(sample_map_c, "b h w d ->  (b h w d)")
        sample_map_flatten[uncer_idx] = 1
        # random sample
        p = torch.ones(b*h*w*d).cuda()
        p[~mask_flatten] = 0
        p[uncer_idx] = 0
        p = p.detach().cpu().numpy()
        p = p / np.linalg.norm(p, ord=1)
        sample_idx = np.random.choice(np.arange(p.shape[0]), num_random_points, p=p)
        sample_idx = torch.from_numpy(sample_idx).long().cuda()
        sample_map_flatten[sample_idx] = 1
        sample_map_c = rearrange(sample_map_flatten, "(b h w d) -> b h w d ", b=b, h=h, w=w)
        sample_map += sample_map_c
    return sample_map


# use topk smaller uncertainty
def cal_smapling_mask_reliability(label,sigma_sq,num_points):
    """

    :param sigma_sq: (B*D,C,H,W)
    :param label: (B,H,W,D)
    :return:
    """
    sigma = rearrange(sigma_sq, "(b d) c h w -> b c h w d ", b=label.size(0))
    sigma_mean = torch.mean(sigma,dim=1) # b h w d
    sample_map = torch.zeros_like(label).float().cuda()
    b, h, w, d = label.size()
    for idx in range(num_classes):
        sample_map_c = torch.zeros_like(label).float().cuda()
        mask = label==idx
        sigma_mean_c = sigma_mean * mask
        sigma_c_flatten = rearrange(sigma_mean_c, "b h w d ->  (b h w d)")
        sigma_map_flatten = rearrange(sample_map_c, "b h w d ->  (b h w d)")
        select_idx_flatten = torch.topk(sigma_c_flatten[sigma_c_flatten!=0], k=num_points, largest=False)[1]
        sigma_map_flatten[select_idx_flatten] = 1
        sample_map_c = rearrange(sigma_c_flatten, "(b h w d) -> b h w d ", b=b, h=h, w=w)
        sample_map += sample_map_c
    return sample_map






"""
from LA_doubleCls_train.py
"""

# Method 2: Coreset
def cal_sampling_mask_coreset_fixNum(label, mu, num_list):
    """
    :param label: (B,H,W,D)
    :param mu: (B*D,dim,H,W)
    :param num_list: the number of coreset for each class. list
    :return:
    """
    mu = rearrange(mu, "(b d) c h w -> b c h w d ", b=label.size(0))
    mu_channel_last = mu.permute(0, 2, 3, 4, 1)  # b h w d c
    b, h, w, d = label.size()
    sample_map = torch.zeros_like(label).float().cuda()
    for idx in range(num_classes):
        num = num_list[idx]
        mask = label == idx
        sample_map_c = torch.zeros_like(label).float().cuda()
        mu_c_mean = torch.mean(mu_channel_last[mask.bool()], dim=0).unsqueeze(0)  # 1,c
        for i in range(b):
            if torch.sum(mask[i]) <= num:
                sample_map_c[i] = mask[i]
                continue
            # dist_sum = torch.sum(1- F.cosine_similarity(mu_channel_last[mask.bool()],mu_c_mean,dim=-1))
            dist_all = 1 - F.cosine_similarity(mu_channel_last[i], mu_c_mean, dim=-1)  # find the most dominant features
            dist_all = torch.clamp(dist_all, 0, 2)
            dist_c = dist_all * mask[i]
            qx_c = 0.5 * 1 / torch.sum(mask[i]) + 0.5 * dist_c / (torch.max(dist_c) + 1e-7)  # b h w d
            qx_c = qx_c * mask[i]
            p = rearrange(qx_c, "h w d ->  (h w d)").detach().cpu().numpy()
            p = p / np.linalg.norm(p, ord=1)
            sample_map_flatten = rearrange(sample_map_c[i], "h w d ->  (h w d)")
            sample_map_flatten_cpu = sample_map_flatten.detach().cpu().numpy()
            sample_idx = np.random.choice(np.arange(sample_map_flatten_cpu.shape[0]), num, p=p, replace=False)
            sample_map_flatten_cpu[sample_idx] = 1
            sample_map_c[i] = rearrange(torch.from_numpy(sample_map_flatten_cpu).cuda(), "(h w d) -> h w d ", h=h, w=w)
        sample_map += sample_map_c
    return sample_map

# Method 3: Hybrid

def cal_sampling_mask_reliability_and_coreset_fixNum(label, mu, sigma_sq, numpoints, coreset_ratio):
    """
    :param label: (B,H,W,D)
    :param mu: (B*D,C,H,W)
    :param sigma_sq: (B*D,C,H,W)
    :param numpoints: the total sampling points
    :param coreset_ratio: the ration of coreset points
    :return:
    """
    coreset_num = int(numpoints * coreset_ratio)
    reliability_num = numpoints - coreset_num
    sample_map_core = cal_sampling_mask_coreset_fixNum(label, mu, [coreset_num, coreset_num])
    sample_map_relia = cal_sampling_mask_reliability_fixNum(label, sigma_sq, reliability_num)
    return torch.logical_or(sample_map_core, sample_map_relia)


# Method 4: NMS
def NMS(feats, mask, budget, threshold):
    """

    :param feats: (B,H,W,D,C) mu of the pixels
    :param mask: (B,H,W,D) class mask
    :param budget: int, the number of sampling pixels
    :param threshold: float, the similarity threshold
    :return:
    """
    b, h, w, d = mask.size()
    # step 1: calculate the cluster the features
    cluster = torch.mean(feats[mask.bool()], dim=0).unsqueeze(0)  # 1,C
    # step 2: calculate the cosine sim between cluster and pixels of the same class
    sim_c = F.cosine_similarity(feats, cluster, dim=-1)  # B,H,W,D
    sim_c[~mask.bool()] = 2  # ensure the other classes can not be retrieved, set the similarity very high
    sim_c_flatten = rearrange(sim_c, "b h w d ->  (b h w d)")
    # step 3: filter out the pixels that share a high degree of similarity with some other and are close to the cluster center.
    sample_map_c = torch.zeros_like(sim_c_flatten)
    sim_order, sim_order_idx = torch.sort(sim_c_flatten)

    if torch.sum(mask) <= budget:
        return mask

    indx = []
    num = 0
    sample_idx = 0

    while num < budget:
        indx.append(sample_idx)
        num += 1
        nextidx = (torch.nonzero(sim_order[sim_order != 2] >= (sim_order[sample_idx] + threshold)).T)[0]
        if nextidx.size(-1) == 0:
            print("not enough for 500")
            break
        sample_idx = nextidx[0]

    # print("final index: {}".format(indx[-1]))

    select_idx = sim_order_idx[torch.tensor(indx).cuda()]
    if len(select_idx) != 0:
        sample_map_c[select_idx] = 1
    sample_map_c = rearrange(sample_map_c, "(b h w d) -> b h w d ", b=b, h=h, w=w)
    return sample_map_c


def cal_sampling_mask_NMS(label, mu, num_classes, budget, threshold):
    """
    :param mu: (B*D,C,H,W)
    :param label: (B,H,W,D)
    :param num_classes: the number of class
    :param budget: the number of sampling pixels
    :param threshold: float, the similarity threshold
    :return:
    """
    sample_map = torch.zeros_like(label).float().cuda()
    mu = rearrange(mu, "(b d) c h w -> b c h w d ", b=label.size(0))
    mu_channel_last = mu.permute(0, 2, 3, 4, 1)  # b h w d c
    for idx in range(num_classes):
        mask = label == idx
        sample_map_c = NMS(mu_channel_last, mask, budget, threshold)
        sample_map += sample_map_c
    return sample_map

# Method 5: Equal Sampling
def cal_sampling_mask_Equal(label):
    sample_map = torch.zeros_like(label).float()
    vol_shape = label.shape
    sampel_nums = []
    for idx in range(num_classes):
        sampel_nums.append(torch.sum(label == idx))
    sample_rate = [min(sampel_nums) / num for num in sampel_nums]
    for idx in range(num_classes):
        prob = 1 - sample_rate[idx]
        rand_map = torch.rand(vol_shape).cuda() * (label == idx)
        rand_map = (rand_map > prob) * 1.0
        sample_map += rand_map
    return sample_map





def reduce_ones(matrix, reduce_count):
    matrix = matrix.detach().cpu().numpy()
    indices = np.argwhere(matrix == 1)  # 获取当前矩阵中1的位置
    num_ones = len(indices)  # 当前矩阵中1的数量
    # 随机选择一些1的位置，并将其设置为0，直到达到指定的减少数量
    indices_to_reduce = np.random.choice(range(num_ones), size=reduce_count, replace=False)
    for index in indices_to_reduce:
        x, y, z = indices[index]
        matrix[x][y][z] = 0
    matrix = torch.from_numpy(matrix).cuda()
    return matrix


def cal_sampling_mask_reliability_fixNum(label, sigma_sq, num):
    """
    :param sigma_sq: (B*D,C,H,W)
    :param label: (B,H,W,D)
    :param num: int: the number of sample points for each class in a single volume
    :return:
    """
    B = label.size(0)
    sigma = rearrange(sigma_sq, "(b d) c h w -> b c h w d ", b=B)
    sigma_channel_last = sigma.permute(0, 2, 3, 4, 1)  # b h w d c
    sample_map = torch.zeros_like(label).cuda()

    for idx in range(num_classes):
        sigma = torch.mean(sigma_channel_last, dim=-1)  # b h w d
        clsmask = (label == idx).bool()
        cls_sigma = sigma[clsmask]
        cls_sigma_np = cls_sigma.detach().cpu().numpy()
        cls_sampleMsk = torch.zeros_like(label).cuda()
        for i in range(B):
            reliableMap = (sigma[i].le(np.percentile(cls_sigma_np, args.thre_weak)) & sigma[i].ge(
                np.percentile(cls_sigma_np, args.thre_strong))) * clsmask[i]
            if torch.sum(reliableMap) <= num:
                cls_sampleMsk[i] = reliableMap
            else:
                # print("before: ",torch.sum(reliableMap))
                cls_sampleMsk[i] = reduce_ones(reliableMap, torch.sum(reliableMap).item() - num)
            # print(torch.sum(cls_sampleMsk[i]))
        sample_map += cls_sampleMsk

    return sample_map