R"""
"""
#
import torch
from typing import List, Tuple
from sklearn.metrics import roc_auc_score, f1_score


#
CE = 0
ERR = 1
MACRO = 2 # ROC-AUC, ovo, macro average
F1 = 3 # F1-score, macro average

CE_REP = 'CE'
ERR_REP = 'ERR'
MACRO_REP = 'ROCAUC'
F1_REP = 'F1'

METRICS_REP = {
    CE: CE_REP,
    ERR: ERR_REP,
    MACRO: MACRO_REP,
    F1: F1_REP,

}

def ce_loss(
    output: torch.Tensor, target: torch.Tensor, weight: torch.Tensor,
    /,
) -> torch.Tensor:
    R"""
    Loss function.
    """
    #
    if not (len(output.shape) == 2 and len(target.shape) == 1):
        # UNEXPECT:
        # Loss function must be computed on batched vectors.
        raise NotImplementedError(
            "Loss function must be computed on batched vectors.",
        )

    #
    return torch.nn.functional.cross_entropy(output, target, weight=weight)


def metrics(
    output: torch.Tensor, target: torch.Tensor, weight: torch.Tensor,
    /,
) -> List[Tuple[int, float]]:
    R"""
    All evaluation metrics.
    """
    #
    if not (len(output.shape) == 2 and len(target.shape) == 1):
        # UNEXPECT:
        # Loss function must be computed on batched vectors.
        raise NotImplementedError(
            "Loss function must be computed on batched vectors.",
        )

    #
    ce = (
        torch.nn.functional.cross_entropy(
            output, target,
            weight=weight, reduction="none",
        )
    )
    
    #
    score = torch.softmax(output, dim=1)
    pred = torch.argmax(score, dim=1)
    err = (pred != target).to(torch.float32)
    # Compute ROC-AUC (macro)
    try:
        #
        if score.shape[1] == 2: 
            score_bin = score[:, 1] # based on the suggestion of sklearn.metrics.roc_auc_score
            # roc_auc_score(target.cpu().numpy(), score_bin.cpu().numpy(), average="macro", multi_class="ovo")
            macro_roc = (
                roc_auc_score(
                    target.cpu().numpy(), score_bin.cpu().numpy(),
                    average="macro", multi_class="ovo",
                )
            )
        else:
            # in case the target does not contain all classes in the batch
            unique_labels = list(sorted(set(target.cpu().numpy())))
            if len(unique_labels) < score.shape[1]:
                if len(unique_labels) < score.shape[1] - 1:
                    print('Warning: missing samples for {} classes'.format(
                        score.shape[1] - len(unique_labels)
                    ))
                
                pseudo_score = torch.softmax(score[:, unique_labels], dim=-1)
                macro_roc = roc_auc_score(target.cpu().numpy(), pseudo_score.cpu().numpy(), average="macro", multi_class="ovo", labels=unique_labels)
            else:
                macro_roc = (
                    roc_auc_score(
                        target.cpu().numpy(), score.cpu().numpy(),
                        average="macro", multi_class="ovo",
                    )
                )

    except Exception as e:
        print('roc_auc error: ', e)        
        #
        macro_roc = 0.0
        import pdb; pdb.set_trace()
    
    
    # Compute F1 scores (macro) 
    if score.shape[1] == 2:
        pred_bin = pred
        macro_f1 = f1_score(
            target.cpu().numpy(), pred_bin.cpu().numpy(),
            average="macro",
        )
    else:
        # in case the target does not contain all classes in the batch
        unique_labels = list(sorted(set(target.cpu().numpy())))
        if len(unique_labels) < score.shape[1]:
            if len(unique_labels) < score.shape[1] - 1:
                print('Warning: missing samples for {} classes'.format(
                    score.shape[1] - len(unique_labels)
                ))
            macro_f1 = f1_score(
                target.cpu().numpy(), pred.cpu().numpy(),
                average="macro", labels=unique_labels
            )
        else:
            macro_f1 = f1_score(
                target.cpu().numpy(), pred.cpu().numpy(),
                average="macro",
            )
    # Return metrics in the same ordered slots as the constants defined above:
    # CE=0, ERR=1, MACRO=2 (roc-auc macro), MICRO=3 (roc-auc micro), MACRO_F1=4, MICRO_F1=5
    return [
        (len(ce), torch.sum(ce).item()),
        (len(err), torch.sum(err).item()),
        (1, -macro_roc),
        (1, -macro_f1)
    ]