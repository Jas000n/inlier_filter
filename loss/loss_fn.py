import torch
import torch.nn.functional as F
import torch
import torch.nn.functional as F

def classification_Loss(pred, gt):
    # pred (torch.Tensor): Predicted logits with shape (b, n, 2)
    # gt (torch.Tensor): Ground truth labels with shape (b, n, 1)
   
    b, n, _ = pred.shape  # batch size, number of elements
    loss = 0
    wrong = 0
    for i in range(b):
        # Compute the weights dynamically based on the ground truth labels
        labels = gt[i].squeeze(-1).long()  # remove the last dimension to match logits shape
        pos_weight = (labels == 1).sum().float() / n
        neg_weight = 1 - pos_weight
        weight_tensor = torch.tensor([neg_weight, pos_weight], device=pred.device)

        # Define the loss function with computed weights
        lossfn = torch.nn.CrossEntropyLoss(weight=weight_tensor)

        # Compute the loss for the current batch
        batch_loss = lossfn(pred[i], labels)
        loss += batch_loss

        # Calculate the accuracy for the current batch
        with torch.no_grad():
            predictions = pred[i].argmax(dim=1)
            wrong += (predictions != labels).sum()

    accuracy = 1 - wrong.float() / (b * n)
    print("cluster_accuracy:{}".format(accuracy))

    return loss / b  # Return the average loss across all batches
