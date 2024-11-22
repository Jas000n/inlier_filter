from sklearn.cluster import KMeans
import time
import numpy as np
import matplotlib.pyplot as plt
def cluster_and_pad(point_cloud, inlier_mask, target_clusters=2048, cluster_size=16):
    n = point_cloud.shape[0] #num of points
    features = point_cloud.shape[1]

    # Step 1: Perform KMeans clustering
    kmeans = KMeans(n_clusters=target_clusters, random_state=0,n_init=3).fit(point_cloud)
    labels = kmeans.labels_

    # Step 2: Determine inliers based on inlier mask
    cluster_inliers = np.zeros(target_clusters)
    for i in range(n):
        cluster_id = labels[i]
        if inlier_mask[i] == 1:
            cluster_inliers[cluster_id] = 1

    # Step 3: Reshape clusters to (2048, 16, 3) and handle padding/removing
    clusters = []
    for cluster_id in range(target_clusters):
        points = point_cloud[labels == cluster_id]
        if len(points) > cluster_size:
            np.random.shuffle(points)
            points = points[:cluster_size]
        elif len(points) < cluster_size:
            num_repeats = cluster_size // len(points)
            new_points = points.copy()
            if num_repeats >= 1:
                for _ in range(num_repeats - 1):
                    new_points = np.concatenate([new_points, points])

            remaining_points_needed = cluster_size - len(new_points)
            if remaining_points_needed > 0:
                additional_indices = np.random.choice(len(points), remaining_points_needed, replace=True)
                additional_points = points[additional_indices]
                new_points = np.concatenate([new_points, additional_points])
            points = new_points

        clusters.append(points)

    # Convert clusters list to numpy array
    clusters = np.array(clusters)

    return clusters, cluster_inliers.reshape(-1, 1),kmeans.cluster_centers_,labels

# Example usage:
# Assuming point_cloud is your numpy array of shape (n, 3)
# and inlier_mask is your inlier mask array of shape (n, 1)
if __name__ =="__main__":
    time1 = time.time()
    point_cloud = np.load("/home/sy3913/kitti/overlap_11-20/0_617_1362.npy", allow_pickle=True).item()["source_point_cloud"]

    inlier_mask = np.random.randint(0, 2, size=(point_cloud.shape[0], 1))  # Example inlier mask

    src_cluster,gt_src_cluster_mask,src_cluster_center = cluster_and_pad(point_cloud, inlier_mask)
    time2 = time.time()
    print(time2-time1)
    print("Clustered shape:", src_cluster.shape)  # Expected (2048, 16, 3)
    print("Cluster inliers shape:", gt_src_cluster_mask.shape)  # Expected (2048, 1)
