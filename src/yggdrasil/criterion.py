import torch
import torch.nn as nn


class RMSEuclidianLoss(nn.Module):
    def __init__(self):
        super(RMSEuclidianLoss, self).__init__()

    def forward(self, emb_anchor, emb_other, y):
        diff = emb_anchor - emb_other
        dist_sq = torch.sum(torch.pow(diff, 2), 1)
        dist = torch.sqrt(dist_sq)
        loss = torch.sum(torch.pow((dist - y), 2), 1)
        return loss


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, emb_anchor, emb_other, y):
        diff = emb_anchor - emb_other
        dist_sq = torch.sum(torch.pow(diff, 2), 1)
        dist = torch.sqrt(dist_sq)

        mdist = self.margin - dist
        dist = torch.clamp(mdist, min=0.0)
        loss = y * dist_sq + (1 - y) * torch.pow(dist, 2)
        loss = torch.sum(loss) / 2.0 / emb_anchor.size()[0]
        return loss


class LosslessTripletLoss(nn.Module):

    def __init__(self, emb_dim, beta, epsilon=1e-8):
        super(LosslessTripletLoss, self).__init__()
        self.emb_dim = emb_dim
        self.beta = beta
        self.epsilon = epsilon

    def forward(self, emb_anchor, emb_pos, emb_neg):
        diff_pos = emb_anchor - emb_pos
        dist_pos = torch.sum(torch.pow(diff_pos, 2), 1)

        diff_neg = emb_anchor - emb_neg
        dist_neg = torch.sum(torch.pow(diff_neg, 2), 1)

        transformed_dist_pos = -torch.log(
            (dist_pos / self.beta) + 1 + self.epsilon)
        transformed_dist_neg = -torch.log(
            (dist_neg / self.beta) + 1 + self.epsilon)

        return dist_pos, dist_neg, transformed_dist_pos, transformed_dist_neg
