import torch
import torch.nn.functional as F
def cal_precision_recall(gt_mask,pred_mask):
    
    # gt_mask = (b,n,1) label in form (0,1), represent it is positive or negative
    # pred_mask = (b,n,2) logits
    with torch.no_grad():
        pred_probs = F.softmax(pred_mask, dim=-1)
        
        # Select the predicted class (1 if the probability of class '1' is higher than class '0')
        pred_class = pred_probs.argmax(dim=-1)
        
        # True Positives (TP): predicted as 1, actual is 1
        TP = (pred_class == 1) & (gt_mask.squeeze(-1) == 1)
        
        # False Positives (FP): predicted as 1, actual is 0
        FP = (pred_class == 1) & (gt_mask.squeeze(-1) == 0)
        
        # False Negatives (FN): predicted as 0, actual is 1
        FN = (pred_class == 0) & (gt_mask.squeeze(-1) == 1)
        
        # Calculate precision and recall
        precision = TP.sum().float() / (TP.sum() + FP.sum()).float()
        recall = TP.sum().float() / (TP.sum() + FN.sum()).float()
        
        # Handle potential division by zero
        precision = precision if not torch.isnan(precision) else torch.tensor(0.0)
        recall = recall if not torch.isnan(recall) else torch.tensor(0.0)
        
        # Calculate F1 score
        f1_score = 2 * (precision * recall) / (precision + recall)
        f1_score = f1_score if not torch.isnan(f1_score) else torch.tensor(0.0)
        
        return precision.item(), recall.item(), f1_score.item()
def calculate_rmse(gt, pred):
    
    if gt.shape != pred.shape:
        raise ValueError("The shape of ground truth data and prediction data must be the same.")

    with torch.no_grad():

        mse = torch.mean((gt - pred) ** 2)
        rmse = torch.sqrt(mse)
    
    return rmse.item()  

if __name__ == "__main__":
    gt_mask = torch.tensor([
    [[1], [0], [1], [0], [1]],
    [[0], [0], [1], [1], [0]],
    [[1], [1], [0], [0], [1]]
    ])
    pred_mask = torch.tensor([
    [[0.1, 2.0], [3.0, -1.0], [0.5, 1.5], [2.5, 0.5], [100.0, 1.0]],
    [[2.5, -0.5], [1.0, 0.0], [-1.0, 1.0], [0.0, 3.0], [1.5, 0.5]],
    [[-0.5, 2.5], [0.0, 1.0], [2.0, -1.0], [3.0, 0.1], [-0.5, 0.5]]
    ])  
    precision, recall,f1 = cal_precision_recall(gt_mask, pred_mask)

    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1: {f1:.2f}")

    gt = torch.tensor([3, 5, 2.5, 1])  # Example ground truth data
    pred = torch.tensor([2.5, 5.0, 2.0, 1.2])  # Example 



    rmse_value = calculate_rmse(gt, pred)
    print("RMSE:", rmse_value)

