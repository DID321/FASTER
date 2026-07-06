import torch.nn as nn
import torch
from scipy.cluster.hierarchy import weighted


def get_mask(labels1, labels2):
    device = (torch.device('cuda')
              if labels1.is_cuda
              else torch.device('cpu'))
    labels1 = labels1.contiguous().view(-1, 1)
    labels2 = labels2.contiguous().view(-1, 1)
    if labels1.shape[0] != labels2.shape[0]:
        raise ValueError('Num of labels does not match num of features')

    return torch.eq(labels1, labels2.T).float().to(device)

def get_pitch_weights(pitch_labels, pos_pitchs, norm_value=50.0):
    device = (torch.device('cuda')
              if pitch_labels.is_cuda
              else torch.device('cpu'))
    pitch_labels = pitch_labels.contiguous().view(-1, 1)
    pos_pitchs = pos_pitchs.contiguous().view(-1, 1)
    if pitch_labels.shape[0] != pos_pitchs.shape[0]:
        raise ValueError('Num of labels does not match num of features')

    # 计算权重矩阵[shape, shape], 每个元素权重值在0和1之间, 越接近0表示正样本之间差异越小, 越接近1表示正样本之间差异越大
    weights = torch.abs(pitch_labels - pos_pitchs.T + 1) / norm_value
    weights = weights.to(device)
    return weights

def get_angle_weights(angle_labels, pos_angles):
    # 使用sin/cos正余弦编码对方位角0~360°进行编码, 保证周期性, 计算权重矩阵
    device = (torch.device('cuda')
              if angle_labels.is_cuda
              else torch.device('cpu'))
    if angle_labels.shape[0] != pos_angles.shape[0]:
        raise ValueError('Num of labels does not match num of features')
    angle_labels = angle_labels.squeeze().float()
    rad_labels = angle_labels * torch.pi / 180.0
    pos_angles = pos_angles.squeeze().float()
    rad_pos = pos_angles * torch.pi / 180.0
    angle_labels = torch.stack([torch.sin(rad_labels), torch.cos(rad_labels)], dim=1)
    pos_angles = torch.stack([torch.sin(rad_pos), torch.cos(rad_pos)], dim=1)
    # 计算权重矩阵[shape, shape], 每个元素权重值在0和1之间, 越接近0表示正样本之间差异越小, 越接近1表示正样本之间差异越大
    # weight = 1 - cos_diff, range [0,2]
    # weights = 1.0 - torch.cosine_similarity(angle_labels.unsqueeze(1), pos_angles.unsqueeze(0), dim=2)
    weights = 1.0 - torch.mm(angle_labels, pos_angles.T)
    # 归一化到0~1之间
    weights = weights / 2.0
    return weights.to(device)

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask
        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-8)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


class SupConWeightLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""

    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConWeightLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None, weights=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # logits_mask = torch.ones_like(mask)
        # logits_mask[0:batch_size, 0:batch_size] = 0
        # logits_mask[batch_size:2 * batch_size, batch_size:2 * batch_size] = 0
        # anchor_dot_contrast = anchor_dot_contrast * logits_mask

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # apply weights
        # 判断weights 和 mask的shape是否匹配 为[bsz, bsz]
        if weights is not None:
            weights_mask = torch.ones_like(mask)
            weights_mask[0:batch_size, 0:batch_size] = 0
            weights_mask[batch_size:2 * batch_size, batch_size:2 * batch_size] = 0
            weights = weights.repeat(anchor_count, contrast_count)
            weights = weights * weights_mask
            if weights.shape[0] != mask.shape[0] or weights.shape[1] != mask.shape[1]:
                raise ValueError('Num of weights does not match num of features')
            # 正样本之间差异越小, weights越小, 正样本之间差异越大, weights越大会被惩罚
            mean_log_prob_pos = (mask * weights * log_prob).sum(1) / ((mask * weights_mask).sum(1) + 1e-8)
            # mean_log_prob_pos = (mask * weights * log_prob).sum(1) / (mask * weights).sum(1)
        else:
            # compute mean of log-likelihood over positive
            mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-8)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss
