import torch
import torch.nn.functional as F
def masked_window_loss(logits,segments,pos_weight=10.,dice_weight=1.):
    """Aggregate valid bins within a window, then let the trainer average windows."""
    bce_sum=logits[0].new_zeros(());intersection=bce_sum.clone();denominator=bce_sum.clone();count=0
    for logit,s in zip(logits,segments):
        target=torch.as_tensor(s.target,dtype=logit.dtype,device=logit.device)
        mask=torch.as_tensor(s.valid,dtype=torch.bool,device=logit.device)
        if not bool(mask.any()):continue
        # Select before BCE/Dice: masked NaNs/large logits cannot contaminate loss.
        selected=logit[mask];truth=target[mask];prob=selected.sigmoid()
        bce_sum=bce_sum+F.binary_cross_entropy_with_logits(selected,truth,pos_weight=selected.new_tensor(pos_weight),reduction='sum')
        intersection=intersection+(prob*truth).sum();denominator=denominator+prob.sum()+truth.sum();count+=int(mask.sum())
    if not count:raise RuntimeError('NO_VALID_TRAINING_BINS')
    return bce_sum/count+dice_weight*(1.-(2.*intersection+1e-6)/(denominator+1e-6))
