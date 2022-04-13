'''
Copyright (c) 2022 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2022-03-26 11:35:46
Email: haimingzhang@link.cuhk.edu.cn
Description: The Constrastive loss adapted from SAT
'''

import torch
import torch.nn.functional as F


def contrastive_loss(feat1, feat2, obj_count, margin=0.1, weight=10., reduction=True):
    sim_losses = 0. if reduction else []
    feat1 = F.normalize(feat1, p=2, dim=-1)
    feat2 = F.normalize(feat2, p=2, dim=-1)

    for b_i in range(feat1.shape[0]):
        feat_2d, feat_3d, num_obj = feat1[b_i,:,:], feat2[b_i,:,:], obj_count[b_i]
        feat_2d, feat_3d = feat_2d[:num_obj,:], feat_3d[:num_obj,:]
        cos_scores = feat_2d.mm(feat_3d.t())
        diagonal = cos_scores.diag().view(feat_2d.size(0), 1)
        d1 = diagonal.expand_as(cos_scores)
        d2 = diagonal.t().expand_as(cos_scores)
        # feat_3d retrieval
        cost_3d = (margin + cos_scores - d1).clamp(min=0)
        # feat2d retrieval
        cost_2d = (margin + cos_scores - d2).clamp(min=0)
        # clear diagonals
        I = (torch.eye(cos_scores.size(0)) > .5)
        cost_3d = cost_3d.masked_fill_(I, 0)
        cost_2d = cost_2d.masked_fill_(I, 0)
        topk = min(3,int(cost_3d.shape[0]))
        cost_3d = (torch.topk(cost_3d, topk, dim=1)[0])
        cost_2d = (torch.topk(cost_2d, topk, dim=0)[0])
        if reduction: 
            batch_loss = torch.sum(cost_3d) + torch.sum(cost_2d)
            sim_losses = sim_losses + batch_loss
        else: 
            batch_loss = torch.mean(cost_3d) + torch.mean(cost_2d)
            sim_losses.append(batch_loss)
    
    if reduction: 
        return weight * sim_losses/(torch.sum(obj_count))
    else:
        return weight * torch.tensor(sim_losses)
