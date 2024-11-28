import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import sys
sys.path.append("/scratch/sy3913/mynet")
from dataset.dataloader import retrieve_dataloader




class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.depth = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.dense = nn.Linear(d_model, d_model)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_length, depth)
        """
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        return x.permute(0, 2, 1, 3)

    def scaled_dot_product_attention(self, q, k, v, mask):
        """Calculate the attention weights. q, k, v must have matching leading dimensions.
        The mask has different shapes depending on its type(padding or look ahead)
        but it must be broadcastable for addition.
        """
        matmul_qk = torch.matmul(q, k.transpose(-2, -1))  # (..., seq_len_q, seq_len_k)
        
        # scale matmul_qk
        dk = torch.tensor(k.shape[-1], dtype=torch.float32, device=q.device)
        scaled_attention_logits = matmul_qk / torch.sqrt(dk)

        # add the mask to the scaled tensor.
        if mask is not None:
            scaled_attention_logits += (mask * -1e9)

        # softmax is applied to the last axis (seq_len_k) so that the scores add up to 1.
        attention_weights = F.softmax(scaled_attention_logits, dim=-1)

        output = torch.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

        return output, attention_weights


    def forward(self, q, k, v, mask):
        batch_size = q.shape[0]

        q = self.split_heads(self.W_q(q), batch_size)
        k = self.split_heads(self.W_k(k), batch_size)
        v = self.split_heads(self.W_v(v), batch_size)

        scaled_attention, attention_weights = self.scaled_dot_product_attention(q, k, v, mask)

        scaled_attention = scaled_attention.permute(0, 2, 1, 3).contiguous()
        concat_attention = scaled_attention.view(batch_size, -1, self.d_model)

        output = self.dense(concat_attention)

        return output
class MLPHead(nn.Module):
    def __init__(self, d_model):
        super(MLPHead, self).__init__()
        # Define the network layers
        self.linear1 = nn.Linear(d_model, 128)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(128, 32)
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(32, 2)
        self.softmax = nn.Softmax(dim=2)
    def forward(self, x):
        # x shape: (b, num_cluster, d_model)
        x = self.linear1(x)
        x = self.relu1(x)

        x = self.linear2(x)
        x = self.relu2(x)

        x = self.linear3(x)
        # Output shape: (b, num_cluster, 2)
        return x
    

                     

class PointNet(nn.Module):
    def __init__(self, emb_dims=512): 
        super(PointNet, self).__init__()
        # Using 2D convolutional layers
        self.conv1 = nn.Conv2d(3, 256, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=1, bias=False)
        self.conv3 = nn.Conv2d(256, 512, kernel_size=1, bias=False)
        self.conv4 = nn.Conv2d(512, emb_dims, kernel_size=1, bias=False)


    def forward(self, x):
        # x = (B, C=3, num_cluster, sample_in_cluster)
        point_wise = F.relu(self.conv1(x))
        point_wise = F.relu(self.conv2(point_wise))
        point_wise = F.relu(self.conv3(point_wise))
        point_wise = F.relu(self.conv4(point_wise))

        # Take the max over the dimension representing samples within each cluster
        global_feature, _ = torch.max(point_wise, dim=-1, keepdim=True)
        global_feature = global_feature.squeeze(-1).permute(0,2,1) # Adjust to output shape: B x num_cluster x emb_dims 
        return global_feature 

class FNet(nn.Module):
    def __init__(self, d_model=512, num_heads=8):
        super(FNet, self).__init__()
        self.feature_extractor = PointNet(d_model)
        self.encoder = MultiHeadAttention(d_model, num_heads)
        self.decoder = MultiHeadAttention(d_model, num_heads)
        self.maskPredHead = MLPHead(d_model)
    def forward(self,batch_dict):
        #retrive data from batch dict
        x = batch_dict["src_cluster"]
        y = batch_dict["tgt_cluster"]
        x = x.permute(0, 3, 1, 2) 
        y = y.permute(0, 3, 1, 2) 
        # tokenize points in src and tgt respectively
            # input: (B,3,num_cluster,sample_in_cluster)
            # output: (B x num_cluster x emb_dims)

        x = self.feature_extractor(x) 
        y = self.feature_extractor(y) 

        # encode relative position information of each cluster within point cloud
            # input: (B x num_cluster x emb_dims)
            # output: (B x num_cluster x emb_dims)
        x = self.encoder(x,x,x,None) 
        y = self.encoder(y,y,y,None)

        # decode high dimentional information 
            # input: (B x num_cluster x emb_dims)
            # output: (B x num_cluster x emb_dims)
        x_mask_feature = self.decoder(x,y,y,None)
        y_mask_feature = self.decoder(y,x,x,None)

        # MLP head for inlier classification
             # input: (B x num_cluster x emb_dims)
             # output: (B x num_cluster x 2)(no softmax applied)
        x = self.maskPredHead(x_mask_feature)
        y = self.maskPredHead(y_mask_feature)

        return x,y


    
if __name__ == "__main__":
    train_dl,_, = retrieve_dataloader("KITTI_2048")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    myNet = FNet().to(device)
    for i, batch in enumerate(train_dl):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        x,y = myNet(batch)
        breakpoint()

   