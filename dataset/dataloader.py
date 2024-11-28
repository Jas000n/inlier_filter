from torch.utils.data import DataLoader, Dataset
import os
import numpy as np
import torch


def stack_up_batch(batch):
    batched_data = {}
    for key in batch[0].keys():
        # 检查当前批次中所有数据项的形状是否一致
        shapes = [d[key].shape for d in batch]
        if len(set(shapes)) == 1:
            # 如果所有项形状一致，则堆叠成一个大的numpy数组
            stacked_array = np.stack([d[key] for d in batch])
            batched_data[key] = torch.from_numpy(stacked_array).float()  # 或者根据需要选择其他dtype
        else:
            # 如果形状不一致，将数据放入列表中
            list_of_arrays = [d[key] for d in batch]
            batched_data[key] = list_of_arrays

    return batched_data

# def stack_up_batch(batch):
#     batched_data = {}
#     for key in batch[0].keys():
#         # 检查当前批次中所有数据项的形状是否一致
#         shapes = [d[key].shape for d in batch]
#         if len(set(shapes)) == 1:
#             # 如果所有项形状一致，则堆叠成一个大的numpy数组
#             stacked_array = np.stack([d[key] for d in batch])
#             batched_data[key] = torch.from_numpy(stacked_array).float()  # 或者根据需要选择其他dtype
#         else:
#             # 如果形状不一致，将数据放入列表中
#             list_of_arrays = [d[key] for d in batch]
#             batched_data[key] = list_of_arrays

#     return batched_data

class clustered_KITTI(Dataset):
    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
        self.data_paths_list = self._retrieve_all_npy_paths(self.dataset_path)
    def _retrieve_all_npy_paths(self,dataset_path):
        file_list = []
        for npy_file in os.listdir(dataset_path):
            file_path = os.path.join(dataset_path, npy_file)
            if os.path.isfile(file_path):
                file_list.append(file_path)
        return file_list
    def __len__(self):
        return len(self.data_paths_list)
    def __getitem__(self, index):

        # load data
        npy_path = self.data_paths_list[index]
        npy_file = np.load(npy_path, allow_pickle=True)
        npy_file = npy_file.item()
        src = npy_file["src"]
        tgt = npy_file["tgt"]
        gt_src_mask = npy_file["gt_src_mask"]
        gt_tgt_mask = npy_file["gt_tgt_mask"]
        gt_transformation = npy_file["gt_transformation"]
        src_cluster = npy_file["src_cluster"]
        tgt_cluster = npy_file["tgt_cluster"]
        gt_src_cluster_mask = npy_file["gt_src_cluster_mask"]
        gt_tgt_cluster_mask = npy_file["gt_tgt_cluster_mask"]
        src_cluster_center = npy_file["src_cluster_center"]
        tgt_cluster_center = npy_file["tgt_cluster_center"]
        src_labels = npy_file["src_labels"]
        tgt_labels = npy_file["tgt_labels"]
    #    info = {
    #         'src': src, #(m,3)
    #         'tgt': tgt,
    #         'gt_src_mask': gt_src_mask.numpy(), #(m,1)
    #         'gt_tgt_mask': gt_tgt_mask.numpy(),
    #         'gt_transformation': M,
    #         'src_cluster':src_cluster, #(2048cluster,32points within each cluster,3xyz)
    #         'tgt_cluster':tgt_cluster,
    #         'gt_src_cluster_mask':gt_src_cluster_mask,#(2048,1)
    #         'gt_tgt_cluster_mask':gt_tgt_cluster_mask,
    #         'src_cluster_center':src_cluster_center, #(2048,3)
    #         'tgt_cluster_center':tgt_cluster_center,
    #         'src_labels':src_labels, #(m,1)
    #         'tgt_labels':tgt_labels,

    #     }
        # construct a dict to return
        data = {}
        data["src"] = src
        data["tgt"] = tgt
        data["gt_src_mask"] = gt_src_mask
        data["gt_tgt_mask"] = gt_tgt_mask
        data["gt_transformation"] = gt_transformation
        data["src_cluster"] = src_cluster
        data["tgt_cluster"] = tgt_cluster
        data["gt_src_cluster_mask"] = gt_src_cluster_mask
        data["gt_tgt_cluster_mask"] = gt_tgt_cluster_mask
        data["src_cluster_center"] = src_cluster_center
        data["tgt_cluster_center"] = tgt_cluster_center
        data["src_labels"] = src_labels #(m,1)
        data["tgt_labels"] = tgt_labels
      
    
        return data
def retrieve_dataloader(dataset_name):
    batch_size = 32
    if dataset_name == "KITTI_2048":
        train_ds = clustered_KITTI("/scratch/sy3913/mynet/dataset/preprocessed_dataset/kitti_2048/train")
        test_ds = clustered_KITTI("/scratch/sy3913/mynet/dataset/preprocessed_dataset/kitti_2048/test")

        train_dl = DataLoader(train_ds,batch_size=batch_size,shuffle=True,
                              num_workers=8,pin_memory=True,drop_last=True,collate_fn=stack_up_batch)
        test_dl = DataLoader(test_ds,batch_size=batch_size,shuffle=True,
                              num_workers=8,pin_memory=True,drop_last=True,collate_fn=stack_up_batch)

        return train_dl,test_dl
    else:
        raise Exception("not implemented")
    
if __name__ == "__main__":
    train_dl,_, = retrieve_dataloader("KITTI_2048")
    for i, batch in enumerate(train_dl):

        breakpoint()