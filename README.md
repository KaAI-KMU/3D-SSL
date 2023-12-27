# 3DIoUMatch-based 3D Semi-Supervised Learning

This repository implements a 3D Semi-Supervised Learning framework based on 3DIoUMatch, leveraging OpenPCDet for point cloud processing. It's designed for advanced research and experiments in 3D object detection.

## Setup

To set up the environment, follow these steps:

1. **Follow [OpenPCDet Setup](https://github.com/open-mmlab/OpenPCDet)**: The setup for this project is identical to that of OpenPCDet. Ensure you follow their guidelines to set up your environment correctly. Additionally, you must set up the KITTI dataset according to the instructions.

2. **Create Split Database Info**: 
   - Use the command below to generate split database information.
   - Example Command: 
     ```bash
     python -m pcdet.datasets.kitti.kitti_dataset_ssl create_split_infos tools/cfgs/dataset_configs/kitti_dataset.yaml 002_1
     ```
   - Note: Splits can be created by adding `.txt` files to `data/kitti/Imageset`, allowing for custom split configurations.

## Training

The training process involves two stages: Pretraining and Semi-Supervised Learning.

1. **Pretraining**:
   - Navigate to the `tools` directory.
   - Run the pretraining script with the specified configuration file and split name.
   - Example Command: 
     ```bash
     python pretrain.py --cfg_file cfgs/kitti_models/pv_rcnn.yaml --split_name 002_1 --repeat 10
     ```

2. **Semi-Supervised Learning**:
   - Continue training in a semi-supervised manner using a pretrained model.
   - Example Command: 
     ```bash
     python pretrain.py --cfg_file cfgs/kitti_models/pv_rcnn.yaml --split_name 002_1 --pretrained_model ../output/kitti_models/pv_rcnn/default/ckpt
     ```
