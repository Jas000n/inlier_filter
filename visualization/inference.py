import torch
import numpy as np
import sys
import os
os.environ["OMP_NUM_THREADS"] = "1"
import open3d as o3d


sys.path.append("/Users/jas0n/PycharmProjects/inlier_filter")

def apply_transformation(points, transformation):
    points_hom = np.hstack((points, np.ones((points.shape[0], 1))))
    transformed_points = (transformation @ points_hom.T).T
    return transformed_points[:, :3]
def inference_dye_visualization(data_path, weight_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(weight_path, map_location=device)

    model.eval()
    data = np.load(data_path, allow_pickle=True).item()
    data = {key: torch.from_numpy(value).unsqueeze(0).to(device) for key, value in data.items()}

    # Ensure data types
    data["src_cluster"] = data["src_cluster"].float()
    data["tgt_cluster"] = data["tgt_cluster"].float()

    x_mask_pred, y_mask_pred = model(data)

    src_cluster_label = data["src_labels"].squeeze(0)
    tgt_cluster_label = data["tgt_labels"].squeeze(0)
    src_gt_point_mask = data["gt_src_mask"].squeeze(0)
    tgt_gt_point_mask = data["gt_tgt_mask"].squeeze(0)

    x_mask_pred = torch.softmax(x_mask_pred.squeeze(0), dim=1)
    y_mask_pred = torch.softmax(y_mask_pred.squeeze(0), dim=1)

    pred_src_point_mask = torch.argmax(x_mask_pred, dim=1)[src_cluster_label].unsqueeze(-1)
    pred_tgt_point_mask = torch.argmax(y_mask_pred, dim=1)[tgt_cluster_label].unsqueeze(-1)

    src = data["src"].squeeze().numpy()
    tgt = data["tgt"].squeeze().numpy()
    gt_transformation = data["gt_transformation"].squeeze(0).numpy()

    # Apply transformation
    src_aligned = apply_transformation(src, gt_transformation)

    # For src point clouds
    src_red_color = [1, 0, 0]  # Red color
    tgt_green_color = [0, 1, 0]  # Green color

    # Visualize Raw
    raw_src_pcd = o3d.geometry.PointCloud()
    raw_src_pcd.points = o3d.utility.Vector3dVector(src_aligned)
    raw_src_pcd.paint_uniform_color(src_red_color)
    raw_tgt_pcd = o3d.geometry.PointCloud()
    raw_tgt_pcd.points = o3d.utility.Vector3dVector(tgt)
    raw_tgt_pcd.paint_uniform_color(tgt_green_color)

    # Visualize Predicted
    pred_src_pcd = o3d.geometry.PointCloud()
    if isinstance(pred_src_point_mask, torch.Tensor):
        pred_src_point_mask = pred_src_point_mask.numpy().astype(bool)
    pred_src_pcd.points = o3d.utility.Vector3dVector(src_aligned[pred_src_point_mask[:, 0]])
    pred_src_pcd.paint_uniform_color(src_red_color)
    pred_tgt_pcd = o3d.geometry.PointCloud()
    if isinstance(pred_tgt_point_mask, torch.Tensor):
        pred_tgt_point_mask = pred_tgt_point_mask.numpy().astype(bool)
    pred_tgt_pcd.points = o3d.utility.Vector3dVector(tgt[pred_tgt_point_mask[:, 0]])
    pred_tgt_pcd.paint_uniform_color(tgt_green_color)

    # Visualize GT Mask
    gt_src_pcd = o3d.geometry.PointCloud()
    if isinstance(src_gt_point_mask, torch.Tensor):
        src_gt_point_mask = src_gt_point_mask.numpy().astype(bool)
    gt_src_pcd.points = o3d.utility.Vector3dVector(src_aligned[src_gt_point_mask[:, 0]])
    gt_src_pcd.paint_uniform_color(src_red_color)
    gt_tgt_pcd = o3d.geometry.PointCloud()
    if isinstance(tgt_gt_point_mask, torch.Tensor):
        tgt_gt_point_mask = tgt_gt_point_mask.numpy().astype(bool)
    gt_tgt_pcd.points = o3d.utility.Vector3dVector(tgt[tgt_gt_point_mask[:, 0]])
    gt_tgt_pcd.paint_uniform_color(tgt_green_color)

    if isinstance(pred_src_point_mask, torch.Tensor):
        pred_src_point_mask = pred_src_point_mask.numpy()
    if isinstance(src_gt_point_mask, torch.Tensor):
        src_gt_point_mask = src_gt_point_mask.numpy()

        # Calculate correct and incorrect predictions
    correct_mask = (pred_src_point_mask.squeeze() == src_gt_point_mask.squeeze())
    incorrect_mask = ~correct_mask

    # Visualize Correct Predictions in Green
    correct_src_pcd = o3d.geometry.PointCloud()
    correct_src_pcd.points = o3d.utility.Vector3dVector(src_aligned[correct_mask])
    correct_src_pcd.paint_uniform_color([0, 1, 0])  # Green color

    # Visualize Incorrect Predictions in Red
    incorrect_src_pcd = o3d.geometry.PointCloud()
    incorrect_src_pcd.points = o3d.utility.Vector3dVector(src_aligned[incorrect_mask])
    incorrect_src_pcd.paint_uniform_color([1, 0, 0])  # Red color

    #qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq
    if isinstance(pred_tgt_point_mask, torch.Tensor):
        pred_tgt_point_mask = pred_tgt_point_mask.numpy()
    if isinstance(tgt_gt_point_mask, torch.Tensor):
        tgt_gt_point_mask = tgt_gt_point_mask.numpy()

        # Calculate correct and incorrect predictions
    correct_mask_tgt = (pred_tgt_point_mask.squeeze() == tgt_gt_point_mask.squeeze())
    incorrect_mask_tgt = ~correct_mask_tgt

    # Visualize Correct Predictions in Green
    correct_tgt_pcd = o3d.geometry.PointCloud()
    correct_tgt_pcd.points = o3d.utility.Vector3dVector(tgt[correct_mask_tgt])
    correct_tgt_pcd.paint_uniform_color([0, 0, 1])  # blue color

    # Visualize Incorrect Predictions in Red
    incorrect_tgt_pcd = o3d.geometry.PointCloud()
    incorrect_tgt_pcd.points = o3d.utility.Vector3dVector(tgt[incorrect_mask_tgt])
    incorrect_tgt_pcd.paint_uniform_color([1, 0, 1])  # Red color

    # visualize uncertainty mask
    x_probs_diff = torch.abs(x_mask_pred[:, 0] - x_mask_pred[:, 1])
    y_probs_diff = torch.abs(y_mask_pred[:, 0] - y_mask_pred[:, 1])

    threshold = 0.5

    uncertain_x_mask = x_probs_diff < threshold
    uncertain_y_mask = y_probs_diff < threshold

    uncertain_pred_src_point_mask = uncertain_x_mask[src_cluster_label].unsqueeze(-1)
    certain_pred_src_point_mask = ~uncertain_pred_src_point_mask
    uncertain_pred_tgt_point_mask = uncertain_y_mask[tgt_cluster_label].unsqueeze(-1)
    certain_pred_tgt_point_mask = ~uncertain_pred_tgt_point_mask

    uncertain_src_pcd = o3d.geometry.PointCloud()
    uncertain_src_pcd.points = o3d.utility.Vector3dVector(src_aligned[uncertain_pred_src_point_mask[:,0]])
    uncertain_src_pcd.paint_uniform_color([1, 0, 0])
    certain_src_pcd = o3d.geometry.PointCloud()
    certain_src_pcd.points = o3d.utility.Vector3dVector(src_aligned[certain_pred_src_point_mask[:,0]])
    certain_src_pcd.paint_uniform_color([0, 1, 0])

    uncertain_tgt_pcd = o3d.geometry.PointCloud()
    uncertain_tgt_pcd.points = o3d.utility.Vector3dVector(tgt[uncertain_pred_tgt_point_mask[:, 0]])
    uncertain_tgt_pcd.paint_uniform_color([1, 0, 1])
    certain_tgt_pcd = o3d.geometry.PointCloud()
    certain_tgt_pcd.points = o3d.utility.Vector3dVector(tgt[certain_pred_tgt_point_mask[:,0]])
    certain_tgt_pcd.paint_uniform_color([0, 0, 1])
    for cases in ["src","tgt","together","correct_incorrect_together", "uncertain_certain"]:
        if cases == "src":
            o3d.visualization.draw_geometries([raw_src_pcd])
            o3d.visualization.draw_geometries([pred_src_pcd])
            o3d.visualization.draw_geometries([gt_src_pcd])
        elif cases == "tgt":
            o3d.visualization.draw_geometries([raw_tgt_pcd])
            o3d.visualization.draw_geometries([pred_tgt_pcd])
            o3d.visualization.draw_geometries([gt_tgt_pcd])
        elif cases == "together":
            o3d.visualization.draw_geometries([raw_src_pcd, raw_tgt_pcd])
            o3d.visualization.draw_geometries([pred_src_pcd, pred_tgt_pcd])
            o3d.visualization.draw_geometries([gt_src_pcd, gt_tgt_pcd])
        elif cases == "correct_incorrect_together":
            o3d.visualization.draw_geometries([correct_src_pcd, incorrect_src_pcd, correct_tgt_pcd,incorrect_tgt_pcd])
        elif cases == "uncertain_certain":
            o3d.visualization.draw_geometries([uncertain_src_pcd,certain_src_pcd,uncertain_tgt_pcd,certain_tgt_pcd])

    print("Inference and visualization completed.")

if __name__ == "__main__":
    # inference_dye_visualization("/Users/jason/PycharmProjects/myNet/data/eval_2048/1_overlap_[0.61650666].npy",
    #                             "/Users/jason/PycharmProjects/myNet/weights/2048_499_0.8446222941080729.pth")

    # something wrong with dataset
    # inference_dye_visualization("/Users/jason/PycharmProjects/myNet/data/eval_512/40_overlap_[0.24286875].npy",
    #                             "/Users/jason/PycharmProjects/myNet/weights/512_445_0.8514811197916666.pth")

    inference_dye_visualization("/Users/jas0n/PycharmProjects/inlier_filter/dataset/preprocessed_dataset/train/410_overlap_[0.68498411].npy",
                                "/Users/jas0n/PycharmProjects/inlier_filter/weights/219_0.6363904999523629.pth")
