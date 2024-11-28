import torch
from torch import nn
from tqdm import tqdm
import sys
sys.path.append("/scratch/sy3913/mynet/")
import torch.optim as optim
from dataset.dataloader import retrieve_dataloader
from loss.loss_fn import classification_Loss
from model.FNet import FNet
from metrics.metrics import cal_precision_recall, calculate_rmse
from utils.plotResult import plot_metrics,plot_classification_accuracy
import torch.nn.functional as F

def train_one_epoch(epoch, model, loader,metrics_list, optimizer):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device) 
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        print("Let's use", torch.cuda.device_count(), "GPUs!")


    total_classification = 0.0
    total_accuracy = 0.0
    for batch_idx, data_batch in enumerate(tqdm(loader, desc="Processing batches")):

        optimizer.zero_grad()  
        data_batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in data_batch.items()}
        gt_x_mask = data_batch['gt_src_cluster_mask']
        gt_y_mask = data_batch['gt_tgt_cluster_mask']


    

        pred_x_mask, pred_y_mask = model(data_batch)
        src_loss = classification_Loss(pred_x_mask,gt_x_mask)
        tgt_loss = classification_Loss(pred_y_mask,gt_y_mask)
        loss = src_loss + tgt_loss
        loss.backward()
        optimizer.step()

       
        probabilities = F.softmax(pred_x_mask, dim=2)
        predicted_classes = torch.argmax(probabilities, dim=2)
        gt_classes = gt_x_mask.squeeze(2)
        accuracy_x = (predicted_classes == gt_classes).float().mean()

        probabilities = F.softmax(pred_y_mask, dim=2)
        predicted_classes = torch.argmax(probabilities, dim=2)
        gt_classes = gt_y_mask.squeeze(2)
        accuracy_y = (predicted_classes == gt_classes).float().mean()
        
        precision_x, recall_x, f1_x = cal_precision_recall(gt_x_mask,pred_x_mask)
        precision_y, recall_y, f1_y = cal_precision_recall(gt_y_mask,pred_y_mask)
        print("{} batch, precision:{}\t{}\n,recall:{}\t{},\nf1:{}\t{}".format(batch_idx,precision_x,precision_y, recall_x,recall_y, f1_x,f1_y))


        total_classification+=loss.item()
        total_accuracy+=((accuracy_x+accuracy_y)/2).item()
    
    epoch_metrics_list = [total_classification / len(loader),total_accuracy/ len(loader)]
    metrics_list.append(epoch_metrics_list)

    print(
    f'Epoch {epoch} complete:'
    f'Average Classification Loss: {total_classification / len(loader):.6f}'
    )
def test_one_epoch(epoch, model, loader,metrics_list):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)  # Move the model to the appropriate device
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    
    model.eval()  # Set the model to evaluation mode



    total_classification = 0.0
    total_accuracy = 0.0
    with torch.no_grad():  # Disable gradient computation
        for batch_idx, data_batch in enumerate(loader):
            data_batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in data_batch.items()}
            gt_x_mask = data_batch['gt_src_cluster_mask']
            gt_y_mask = data_batch['gt_tgt_cluster_mask']

    
      
            pred_x_mask, pred_y_mask = model(data_batch)
            src_loss = classification_Loss(pred_x_mask,gt_x_mask)
            tgt_loss = classification_Loss(pred_y_mask,gt_y_mask)
            loss = src_loss + tgt_loss
           


            probabilities = F.softmax(pred_x_mask, dim=2)
            predicted_classes = torch.argmax(probabilities, dim=2)
            gt_classes = gt_x_mask.squeeze(2)
            accuracy_x = (predicted_classes == gt_classes).float().mean()

            probabilities = F.softmax(pred_y_mask, dim=2)
            predicted_classes = torch.argmax(probabilities, dim=2)
            gt_classes = gt_y_mask.squeeze(2)
            accuracy_y = (predicted_classes == gt_classes).float().mean()
        
            precision_x, recall_x, f1_x = cal_precision_recall(gt_x_mask,pred_x_mask)
            precision_y, recall_y, f1_y = cal_precision_recall(gt_y_mask,pred_y_mask)
            print("{} batch, precision:{}\t{}\n,recall:{}\t{},\nf1:{}\t{}".format(batch_idx,precision_x,precision_y, recall_x,recall_y, f1_x,f1_y))


            total_classification+=loss.item()
            total_accuracy+=((accuracy_x+accuracy_y)/2).item()


        epoch_metrics_list = [total_classification / len(loader),total_accuracy/ len(loader)]
        metrics_list.append(epoch_metrics_list)


    print(
    f'Epoch {epoch} complete:'
    f'Average Classification Loss: {total_classification / len(loader):.6f}'
    )

    
if __name__ == '__main__':
    train_dl, test_dl = retrieve_dataloader("KITTI_2048")
    model = FNet()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train_metrics_list=[]
    test_metrics_list=[]
    for i in range(500):
        train_one_epoch(i, model, train_dl,train_metrics_list, optimizer)
        test_one_epoch(i,model,test_dl,test_metrics_list)
        plot_classification_accuracy(train_metrics_list,test_metrics_list,"./result.png")
        torch.save(model, '/scratch/sy3913/mynet/weights/cluster_2048/{}_{}.pth'.format(i,test_metrics_list[i][1]))
