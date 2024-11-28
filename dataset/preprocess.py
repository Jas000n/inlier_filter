# preprocess all dataset, basically do K-Means on them
import os
import random
import numpy as np
import torch
from tqdm import tqdm

import sys
sys.path.append('..') 
from utils.transformation import numpyPCTransformation 
from utils.KMeans import cluster_and_pad



def processData(pcd_path,set:str,threshold=2):
    print("Start processing {} set, number of pairs:{}".format(set,len(pcd_path)))
    random.shuffle(pcd_path)

    for idx, path in tqdm(enumerate(pcd_path), total=len(pcd_path)):
        data = np.load(os.path.join(root_dir, path), allow_pickle=True).item()
        retrieved_src = data['source_point_cloud']
        retrieved_tgt = data['target_point_cloud']
        retrieved_M = data['transformation_matrix']
        src, tgt, M = retrieved_src,retrieved_tgt,retrieved_M
        t_src = numpyPCTransformation(src,M)
        t_src_torch = torch.tensor(t_src).to(device)
        tgt_torch = torch.tensor(tgt).to(device)

        l2 = torch.cdist(t_src_torch, tgt_torch)
        overlap = torch.nonzero(l2 <= threshold)
        gt_src_mask = torch.zeros(t_src_torch.shape[0], 1)
        gt_tgt_mask = torch.zeros(tgt_torch.shape[0], 1)
        gt_src_mask[overlap[:, 0], 0] = 1
        gt_tgt_mask[overlap[:, 1], 0] = 1
        src_cluster,gt_src_cluster_mask,src_cluster_center,src_labels = cluster_and_pad(src,gt_src_mask.numpy())
        tgt_cluster,gt_tgt_cluster_mask,tgt_cluster_center,tgt_labels = cluster_and_pad(tgt,gt_tgt_mask.numpy())
        src_overlappingRation = sum(gt_src_mask==1).numpy()/src.shape[0]
        tgt_overlappingRation = sum(gt_tgt_mask==1).numpy()/tgt.shape[0]
        if(src_overlappingRation<0.1 or tgt_overlappingRation<0.1):
            continue
        dir = "./preprocessed_dataset/kitti_2048/{}".format(set)
        if not os.path.exists(dir):
            os.makedirs(dir)

        info = {
            'src': src, #(m,3)
            'tgt': tgt,
            'gt_src_mask': gt_src_mask.numpy(), #(m,1)
            'gt_tgt_mask': gt_tgt_mask.numpy(),
            'gt_transformation': M,
            'src_cluster':src_cluster, #(2048cluster,32points within each cluster,3xyz)
            'tgt_cluster':tgt_cluster,
            'gt_src_cluster_mask':gt_src_cluster_mask,#(2048,1)
            'gt_tgt_cluster_mask':gt_tgt_cluster_mask,
            'src_cluster_center':src_cluster_center, #(2048,3)
            'tgt_cluster_center':tgt_cluster_center,
            'src_labels':src_labels, #(m,1)
            'tgt_labels':tgt_labels,

        }
        save_path = os.path.join(dir,"{}_overlap_{}.npy".format(idx,src_overlappingRation))
        np.save(save_path, info)
    print("Finish processing training data")
if __name__ =="__main__":
    random.seed(2024)

    if torch.cuda.is_available():
        dev = "cuda:0"
    else:
        dev = "cpu"
    device = torch.device(dev)

    root_dir = '/scratch/wl2454/Low_overlap_PCR/KITTI_dataset/new_dataset/train'
    overlap_ratio = ['0-10','10-20','20-30','30-40','40-50','50-60','60-70','70-80','80-90','90-100']

    train_pcd_path = []
    test_pcd_path = []
    for overlap in overlap_ratio:
        overlap_dir = os.path.join(root_dir, 'overlap_' + overlap)
        tmp_dirs = []
        for file_name in os.listdir(overlap_dir):
            tmp_dirs.append(os.path.join(overlap_dir, file_name))
        random.shuffle(tmp_dirs)
        tmp_train_dir = tmp_dirs[:400]
       
        train_pcd_path.extend(tmp_train_dir)
        
   
    processData(train_pcd_path,"train")