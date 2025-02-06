import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def supervised_contrastive_loss(features, labels, device, temperature=0.05):
    """
    Computes supervised contrastive loss as in:
    "Supervised Contrastive Learning" (Khosla et al.).
    """
    features = F.normalize(features, p=2, dim=1)
    batch_size = features.shape[0]
    mask = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0)).to(device)
    logits = torch.div(torch.matmul(features, features.T), temperature)
    logits_max, _ = torch.max(logits, dim=1, keepdim=True)
    logits = logits - logits_max.detach()
    logits_mask = torch.ones_like(mask, dtype=torch.bool).fill_diagonal_(False)
    mask = mask * logits_mask
    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)
    mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-8)
    loss = -mean_log_prob_pos.mean()
    return loss


def arcface_loss(features, labels, weight, device, margin = 0.3, scale = 32.0):
    """
    Computes ArcFace loss as in:
    "ArcFace: Additive Angular Margin Loss for Deep Face Recognition" (Deng et al.).
    """
    if margin > 1.0:
        raise ValueError("Margin should be less than or equal to 1.0")
    x_norm = F.normalize(features, p=2, dim=1, eps=1e-7)
    w_norm = F.normalize(weight, p=2, dim=1, eps=1e-7)
    cosine = torch.matmul(x_norm, w_norm.t()).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    sine = torch.sqrt(1.0 - cosine ** 2)
    cos_m = math.cos(margin)
    sin_m = math.sin(margin)
    threshold = math.cos(math.pi - margin)
    mm = math.sin(math.pi - margin) * margin
    phi = cosine * cos_m - sine * sin_m
    phi = torch.where(cosine > threshold, phi, cosine - mm)
    one_hot = torch.zeros_like(cosine, device=device)
    one_hot.scatter_(1, labels.view(-1, 1), 1.0)
    phi = torch.clamp(phi, -1.0 + 1e-7, 1.0 - 1e-7)
    margin_logits = torch.where(one_hot == 1, phi, cosine) * scale
    loss = F.cross_entropy(margin_logits, labels)
    return loss


def cosface_loss(features, labels, weight, device, margin = 0.25, scale = 32.0):
    """
    Computes CosFace loss as in:
    "CosFace: Large Margin Cosine Loss for Deep Face Recognition" (Wang et al.).
    """
    if margin > 1.0:
        raise ValueError("Margin should be less than or equal to 1.0")
    x_norm = F.normalize(features, p=2, dim=1, eps=1e-7)
    w_norm = F.normalize(weight, p=2, dim=1, eps=1e-7)
    logits = torch.matmul(x_norm, w_norm.t())
    batch_indices = torch.arange(features.size(0), dtype=torch.long, device=device)
    logits[batch_indices, labels] -= margin
    logits = logits.clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    margin_logits = logits * scale
    loss = F.cross_entropy(margin_logits, labels)
    return loss

