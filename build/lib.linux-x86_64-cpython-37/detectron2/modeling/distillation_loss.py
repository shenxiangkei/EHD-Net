import torch
import torch.nn.functional as F
from fvcore.nn import smooth_l1_loss
from detectron2.distiller_zoo import *

def conloss (anchor_features, contrast_feature, anchor_labels, contrast_labels, P=None,temperature=0.7,loss_weight=0.01):
        """
        Args:
            achor_features: hidden vector of shape [bsz, 1, 256].
            contrast_features: hidden vector of shape [bsz_prime, 1, 256].
            anchor_labels: ground truth of shape [bsz].
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if anchor_features.is_cuda
                  else torch.device('cpu'))

        # if len(anchor_features.shape) < 3:
        #     anchor_features=anchor_features.unsqueeze(1)
        # if len(contrast_feature.shape) < 3:
        #     contrast_feature=contrast_feature.unsqueeze(1)
        # if len(anchor_features.shape) > 3:
        #     features = anchor_features.view(anchor_features.shape[0], anchor_features.shape[1], -1)

        anchor_labels = anchor_labels.view(-1, 1)
        contrast_labels = contrast_labels.view(-1, 1)

        batch_size = anchor_features.shape[0]
        R = torch.eq(anchor_labels, contrast_labels.T).float().requires_grad_(False).to(device)
        mask_p = R.clone().requires_grad_(False)
        mask_p[:, :batch_size] -= torch.eye(batch_size).to(device)
        mask_p = mask_p.detach()
        mask_n = 1 - R
        mask_n = mask_n.detach()


        anchor_dot_contrast = torch.div(
            torch.mm(anchor_features, contrast_feature.T),
            temperature)
        neg_contrast = (torch.exp(anchor_dot_contrast) * mask_n).sum(dim=1,keepdim=True)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        anchor_dot_contrast = anchor_dot_contrast - logits_max.detach()
        if P is None:
            pos_contrast = torch.log(torch.exp(anchor_dot_contrast)) * mask_p - torch.log(
                torch.exp(anchor_dot_contrast) + neg_contrast) * mask_p
        else:
            pos_contrast = torch.log(torch.exp(anchor_dot_contrast)) * mask_p * P - torch.log(
                torch.exp(anchor_dot_contrast) + neg_contrast) * mask_p * P

        num = mask_p.sum(dim=1)
        if len((torch.where(num!=0))[0])==0:
            loss=torch.tensor([0.]).cuda()
        else:
            loss = -torch.div(pos_contrast.sum(dim=1)[num != 0], num[num != 0])
        return {"loss_con": loss_weight*loss.mean()}


def rpn_loss(pred_objectness_logits, pred_anchor_deltas, prev_pred_objectness_logits, prev_pred_anchor_deltas):
    loss = logit_distillation(pred_objectness_logits[0], prev_pred_objectness_logits[0])
    loss += anchor_delta_distillation(pred_anchor_deltas[0], prev_pred_anchor_deltas[0])
    return {"loss_dist_rpn": loss}


def backbone_loss(features, prev_features,param):
    # loss = feature_distillation(features['res4'], prev_features['res4'])
    # loss = feature_distillation(features['p3'], prev_features['p3'], param['p3'])
    # loss = feature_distillation(features['p4'], prev_features['p4'],param['p4'])
    loss = feature_distillation(features['p5'], prev_features['p5'],param['p5'])
    return {"loss_dist_backbone": loss}


def roi_head_loss(pred_class_logits, pred_proposal_deltas, prev_pred_class_logits, prev_pred_proposal_deltas, dist_loss_weight=0.5):
    loss = logit_distillation(pred_class_logits, prev_pred_class_logits)
    # loss = feature_distillation(pred_class_logits, prev_pred_class_logits)
    loss += anchor_delta_distillation(pred_proposal_deltas, prev_pred_proposal_deltas)
    return {"loss_dist_roi_head": dist_loss_weight * loss}


def logit_distillation(current_logits, prev_logits, T=6.0):
    p = F.log_softmax(current_logits / T, dim=1)
    q = F.softmax(prev_logits / T, dim=1)
    kl_div = torch.sum(F.kl_div(p, q, reduction='none').clamp(min=0.0) * (T**2)) / current_logits.shape[0]
    return kl_div


def anchor_delta_distillation(current_delta, prev_delta):
    # return smooth_l1_loss(current_delta, prev_delta, beta=0.1, reduction='mean')
    return F.mse_loss(current_delta, prev_delta)


def feature_distillation(features, prev_features,param):
    # return smooth_l1_loss(features, prev_features, beta=0.1, reduction='mean')
    """
    improved adaptative
    """
    # index=torch.where(param>0.6)
    # old_features=prev_features[index]
    # new_features=features[index]
    new_features=features
    old_features=prev_features
    return F.mse_loss(new_features, old_features)
    # criterion_kd = Attention()
    # criterion_kd = NSTLoss()
    # criterion_kd = DistillKL(T=4)
    # # criterion_kd = FactorTransfer()
    # loss = criterion_kd(features, prev_features)
    # # loss = torch.stack(loss, dim=0).sum()
    # return loss
