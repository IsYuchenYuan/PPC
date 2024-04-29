import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter


# class Uncertainty_head(nn.Module):  # feature -> sigma^2
#     def __init__(self, in_feat=64, out_feat=256,sigma_mode='radius',sigma_trans_mode='sigmoid'):
#         super(Uncertainty_head, self).__init__()
#         if sigma_mode == 'radius':
#             out_feat = 1
#         self.sigma_trans_mode = sigma_trans_mode
#         self.fc1 = Parameter(torch.Tensor(out_feat, in_feat))
#         self.bn1 = nn.BatchNorm2d(out_feat, affine=True)
#         self.relu = nn.ReLU()
#         self.fc2 = Parameter(torch.Tensor(out_feat, out_feat))
#         self.bn2 = nn.BatchNorm2d(out_feat, affine=False)
#         self.gamma = Parameter(torch.Tensor([1.0]))
#         self.beta = Parameter(torch.Tensor([0.0]))
#         # self.gamma = Parameter(torch.Tensor([1e-4])) # 1e-4
#         # self.beta = Parameter(torch.Tensor([-7.0]))
#
#         nn.init.kaiming_normal_(self.fc1)
#         nn.init.kaiming_normal_(self.fc2)
#
#
#
#     def forward(self, x: torch.Tensor):
#         x = x.permute(0, 2, 3, 1)  # B H W C
#         x = F.linear(x, F.normalize(self.fc1, dim=-1))  # [B, W, H, C]
#         x = x.permute(0, 3, 1, 2)  # [B, W, H, C] -> [B, C, W, H]
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = x.permute(0, 2, 3, 1)
#         x = F.linear(x, F.normalize(self.fc2, dim=-1))
#         x = x.permute(0, 3, 1, 2) # [B, W, H, D] -> [B, D, W, H]
#         x = self.bn2(x)
#         # x = self.gamma * (x / self.div) + self.beta
#         x = self.gamma * x + self.beta
#         sigma_sq = torch.log(torch.exp(x) + 1e-6)
#         # log_sigma_sq = torch.log(torch.exp(x) + 1e-6)
#         # return log_sigma_sq
#         if self.sigma_trans_mode == 'sigmoid':
#             sigma_sq = torch.sigmoid(sigma_sq)
#         elif self.sigma_trans_mode == 'softplus':
#             sigma_sq = F.softplus(sigma_sq)
#
#         return sigma_sq



class Uncertainty_head(nn.Module):  # feature -> sigma^2
    def __init__(self, in_feat=64, out_feat=256,sigma_mode='radius',sigma_trans_mode='sigmoid'):
        super(Uncertainty_head, self).__init__()
        if sigma_mode == 'radius':
            out_feat = 1
        self.sigma_trans_mode = sigma_trans_mode
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_feat, out_feat, (3, 3), padding=(1, 1)),
            nn.BatchNorm2d(out_feat),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_feat, out_feat, (1, 1))
        )
        self.fc1 = Parameter(torch.Tensor(out_feat, out_feat))
        self.bn1 = nn.BatchNorm2d(out_feat, affine=True)
        self.relu = nn.ReLU()
        self.fc2 = Parameter(torch.Tensor(out_feat, out_feat))
        self.bn2 = nn.BatchNorm2d(out_feat, affine=False)
        self.gamma = Parameter(torch.Tensor([1.0]))
        self.beta = Parameter(torch.Tensor([0.0]))
        # self.gamma = Parameter(torch.Tensor([1e-4])) # 1e-4
        # self.beta = Parameter(torch.Tensor([-7.0]))

        nn.init.kaiming_normal_(self.fc1)
        nn.init.kaiming_normal_(self.fc2)



    def forward(self, x: torch.Tensor):
        x = self.conv1(x)
        x = x.permute(0, 2, 3, 1)  # B H W C
        x = F.linear(x, F.normalize(self.fc1, dim=-1))  # [B, W, H, C]
        x = x.permute(0, 3, 1, 2)  # [B, W, H, C] -> [B, C, W, H]
        x = self.bn1(x)
        x = self.relu(x)
        x = x.permute(0, 2, 3, 1)
        x = F.linear(x, F.normalize(self.fc2, dim=-1))
        x = x.permute(0, 3, 1, 2) # [B, W, H, D] -> [B, D, W, H]
        x = self.bn2(x)
        # x = self.gamma * (x / self.div) + self.beta
        x = self.gamma * x + self.beta
        sigma_sq = torch.log(torch.exp(x) + 1e-6)
        # log_sigma_sq = torch.log(torch.exp(x) + 1e-6)
        # return log_sigma_sq
        if self.sigma_trans_mode == 'sigmoid':
            sigma_sq = torch.sigmoid(sigma_sq)
        elif self.sigma_trans_mode == 'softplus':
            sigma_sq = F.softplus(sigma_sq)

        return sigma_sq

class Uncertainty_head_inverse(nn.Module):  # feature -> 1 / sigma^2
    def __init__(self, in_feat=64, out_feat=256, sigma_mode="diagonal", sigma_trans_mode='softplusLearn'):
        super(Uncertainty_head_inverse, self).__init__()
        if sigma_mode == 'radius':
            out_feat = 1

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_feat, out_feat, (3, 3), padding=(1, 1)),
            nn.BatchNorm2d(out_feat),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_feat, out_feat, (1, 1))
        )
        self.fc1 = Parameter(torch.Tensor(out_feat, out_feat))
        self.bn1 = nn.BatchNorm2d(out_feat, affine=True)
        self.relu = nn.ReLU()
        self.fc2 = Parameter(torch.Tensor(out_feat, out_feat))
        self.bn2 = nn.BatchNorm2d(out_feat, affine=False)

        self.sigma_transform_mode = sigma_trans_mode


        nn.init.kaiming_normal_(self.fc1)
        nn.init.kaiming_normal_(self.fc2)

        if self.sigma_transform_mode == 'softplusLearn':
            self.scale = Parameter(torch.Tensor([1.0]))
            self.offset = Parameter(torch.Tensor([1.0]))
            self.div = Parameter(torch.Tensor([1.0]))
        elif self.sigma_transform_mode == 'sigmoidLearn':
            self.scale = Parameter(torch.Tensor([4.0]))
            self.offset = Parameter(torch.Tensor([1.0]))

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)
        x = x.permute(0, 2, 3, 1)
        x = F.linear(x, F.normalize(self.fc1, dim=-1))  # [B, W, H, D]
        x = x.permute(0, 3, 1, 2)  # [B, W, H, D] -> [B, D, W, H]
        x = self.bn1(x)
        x = self.relu(x)
        x = x.permute(0, 2, 3, 1)
        x = F.linear(x, F.normalize(self.fc2, dim=-1))
        x = x.permute(0, 3, 1, 2) # [B, W, H, C] -> [B, C, W, H]
        sigma_raw = self.bn2(x)
        if self.sigma_transform_mode == "softplusNarrow":
            offset = 1.0
            scale = 1.0
            sigma = offset + scale * F.softplus(sigma_raw)

        # v8 - basic stuff in a narrow range with sigmoid
        elif self.sigma_transform_mode == "sigmoidNarrow":
            offset = 1.0
            scale = 1.0
            sigma = offset + scale * F.sigmoid(sigma_raw)

        # v9 -- basic, wider range
        elif self.sigma_transform_mode == "sigmoidWide":
            offset = 1.0
            scale = 4.0
            sigma = offset + scale * F.sigmoid(sigma_raw)

        # v10 -- learnable
        elif self.sigma_transform_mode == "softplusLearn":
            sigma =  self.offset + self.scale * F.softplus(sigma_raw / self.div)

        # v10 -- learnable
        elif self.sigma_transform_mode == "sigmoidLearn":
            sigma = self.offset + self.scale * F.sigmoid(sigma_raw)
#
        return sigma # the inverse of sigma_sq