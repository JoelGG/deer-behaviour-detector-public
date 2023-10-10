import torch
import numpy as np


def top_k_accuracy(output: torch.Tensor, target: torch.Tensor, topk=(3,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, y_pred = output.topk(k=maxk, dim=1)
        y_pred = y_pred.t()
        target_reshaped = target.view(1, -1).expand_as(y_pred)
        correct = y_pred == target_reshaped

        list_topk_accs = []
        for k in topk:
            ind_which_topk_matched_truth = correct[:k]
            flattened_indicator_which_topk_matched_truth = (
                ind_which_topk_matched_truth.reshape(-1).float()
            )
            tot_correct_topk = flattened_indicator_which_topk_matched_truth.float().sum(
                dim=0, keepdim=True
            )
            topk_acc = tot_correct_topk / batch_size
            list_topk_accs.append(topk_acc)
        return list_topk_accs


def compute_accuracy(
    labels: Union[torch.Tensor, np.ndarray], preds: Union[torch.Tensor, np.ndarray]
) -> float:
    """
    Args:
        labels: ``(batch_size, class_count)`` tensor or array containing example labels
        preds: ``(batch_size, class_count)`` tensor or array containing model prediction
    """
    assert len(labels) == len(preds)
    return float((labels == preds).sum()) / len(labels)


def compute_class_accuracy(
    labels: Union[torch.Tensor, np.ndarray],
    preds: Union[torch.Tensor, np.ndarray],
    class_index: int,
) -> float:
    """
    Args:
        labels: ``(batch_size, class_count)`` tensor or array containing example labels
        preds: ``(batch_size, class_count)`` tensor or array containing model prediction
        class_index ``class_index`` index of class for accuracy in range 0-9
    """
    assert len(labels) == len(preds)
    idx = labels == class_index
    return float((labels[idx] == preds[idx]).sum() / len(labels[idx]))


def compute_average_class_accuracy(
    labels: Union[torch.Tensor, np.ndarray],
    preds: Union[torch.Tensor, np.ndarray],
    class_list: Union[torch.Tensor, np.ndarray],
) -> float:
    """
    Args:
        labels: ``(batch_size, class_count)`` tensor or array containing example labels
        preds: ``(batch_size, class_count)`` tensor or array containing model prediction
    """
    assert len(labels) == len(preds)
    accuracy = np.zeros(shape=len(class_list))
    for c in class_list:
        accuracy[c] = compute_class_accuracy(labels, preds, c)
    return accuracy.mean()
