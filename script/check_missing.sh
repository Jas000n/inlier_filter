#!/bin/bash

# 设定文件夹路径
directory_path='/scratch/sy3913/mynet/dataset/preprocessed_dataset/KITTI_2048/test'

# 循环检查每个编号的文件是否存在
for i in {0..1499}
do
  # 构建文件名模式
  file_pattern="${directory_path}/${i}_overlap_*.npy"

  # 检查是否存在这样的文件
  if ! ls $file_pattern 1> /dev/null 2>&1; then
    # 如果文件不存在，打印缺失的文件编号
    echo "Missing file for index: $i"
  fi
done
