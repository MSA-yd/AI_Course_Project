import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class FocalLoss(nn.Module):
    def __init__(self, alpha, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, output, target):
        logp = F.cross_entropy(output, target, weight=self.alpha, reduction='none')
        p = torch.exp(-logp)
        focal_loss = (1 - p) ** self.gamma * logp
        return focal_loss.mean()

class LDAMLoss(nn.Module):
    def __init__(self, cls_num_list, device, max_m=0.5, weight=None, s=30):
        super(LDAMLoss, self).__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list.cpu().numpy()))
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.tensor(m_list, dtype=torch.float32).to(device)
        self.m_list = m_list
        self.device = device
        self.s = s
        self.weight = weight

    def forward(self, x, target):
        index = torch.zeros_like(x, dtype=torch.bool)
        index.scatter_(1, target.data.view(-1, 1), True)
        index_float = index.type(torch.float32).to(self.device)
        batch_m = torch.matmul(self.m_list[None, :], index_float.t())
        batch_m = batch_m.view(-1, 1)
        x_m = x - batch_m
        output = torch.where(index, x_m, x)
        return F.cross_entropy(self.s * output, target, weight=self.weight)

class LMFLoss(nn.Module):
    def __init__(self, cls_num_list, device, weight, alpha=0.2, beta=0.2, gamma=2, max_m=0.8, s=5, add_LDAM_weigth=False): 
        super().__init__()
        self.focal_loss = FocalLoss(alpha=weight, gamma=gamma)
        LDAM_weight = weight if add_LDAM_weigth else None
        self.ldam_loss = LDAMLoss(cls_num_list, device, max_m=max_m, weight=LDAM_weight, s=s)
        self.alpha = alpha
        self.beta = beta

    def forward(self, output, target):
        focal_loss_output = self.focal_loss(output, target)
        ldam_loss_output = self.ldam_loss(output, target)
        total_loss = self.alpha * focal_loss_output + self.beta * ldam_loss_output
        return total_loss