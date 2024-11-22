import numpy as np
import torch
def numpyPCTransformation(src,M):
    # do transformation on numpy, src (n,3) M(4*4)
    rot = M[:3, :3]
    trans = M[:3, 3][:, None]
    t_src = ((rot @ src.T) + trans).T
    return t_src
def torchPCTransformation(src, M):

    #do transformation on tensor, src (b,n,3) M(b,4*4)
    rot = M[:, :3, :3]  
    trans = M[:, :3, 3].unsqueeze(2)  
    rotated_src = torch.bmm(src, rot.transpose(1, 2))

    # Ensure trans is broadcastable to the shape of rotated_src
    #ignore trans for now
    trans_broadcasted = trans.expand(-1, -1, src.size(1))  # Explicitly broadcast trans to match src shape

    # Add the translation to the rotated source points
    t_src = rotated_src 

    return t_src

def create_random_transformation(b):
    # Generate random rotation angles for each transformation in the batch
    theta_z = np.random.uniform(-np.pi, np.pi, size=(b,))
    
    # Initialize an array to hold all transformation matrices
    Rz_batch = np.zeros((b, 4, 4),dtype=np.float32)
    
    # Fill the transformation matrices
    for i in range(b):
        cos_z = np.cos(theta_z[i])
        sin_z = np.sin(theta_z[i])
        
        Rz_batch[i] = np.array([
            [cos_z, -sin_z, 0, 0],
            [sin_z, cos_z, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
    
    # Convert the numpy array to a PyTorch tensor
    return torch.from_numpy(Rz_batch)

import numpy as np

def random_rotation_around_z(batch):
    # Generating a random angle for rotation around the z-axis for each point cloud in the batch
    angles = torch.rand(batch.size(0)) * 2 * torch.pi
    
    # Rotation matrix for z-axis
    cos_angles = torch.cos(angles)
    sin_angles = torch.sin(angles)
    
    # Constructing the rotation matrix for each point cloud in the batch
    zeros = torch.zeros_like(cos_angles)
    ones = torch.ones_like(cos_angles)
    rotation_matrices = torch.stack([
        cos_angles, -sin_angles, zeros,
        sin_angles, cos_angles, zeros,
        zeros, zeros, ones
    ], dim=-1).view(-1, 3, 3).to(batch.device)
    
    # Applying the rotation to each point in each cluster of each point cloud
    # Using torch.matmul for batch matrix multiplication: (b, 2048, 16, 3) @ (b, 3, 3) -> (b, 2048, 16, 3)
    rotated = torch.matmul(batch, rotation_matrices.unsqueeze(1))
    return rotated



if __name__ == "__main__":
    # Example point cloud data
    b, num_clusters, points_per_cluster, dims = 4, 2048, 16, 3  # simplified dimensions for demonstration
    example_point_cloud = np.random.rand(b, num_clusters, points_per_cluster, dims)

    # Perform random rotation around Z-axis
    rotated_point_cloud = random_rotation_around_z(example_point_cloud)
    print(rotated_point_cloud)
    breakpoint()    
