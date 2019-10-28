import copy
from pathlib import Path
import pickle

import fire

import second.data.kitti_dataset as kitti_ds
import second.data.nuscenes_dataset as nu_ds
from second.data.all_dataset import create_groundtruth_database

def kitti_data_prep(root_path):
    kitti_ds.create_kitti_info_file(root_path)
    kitti_ds.create_reduced_point_cloud(root_path)
    create_groundtruth_database("KittiDataset", root_path, Path(root_path) / "kitti_infos_train.pkl")

def nuscenes_data_prep(root_path, version, dataset_name, max_sweeps=10):
    nu_ds.create_nuscenes_infos(root_path, version=version, max_sweeps=max_sweeps)
    name = "infos_train.pkl"
    phase='train'
    if version == "v1.0-test":
        name = "infos_test.pkl"
        phase='test'
    if phase=='train':
        create_groundtruth_database(dataset_name,root_path, Path(root_path) / name)

if __name__ == '__main__':
    #kitti_data_prep('/home/zp/data/KITTI')
    nuscenes_data_prep('/home/zp/data/nuscene/nuscene-mini/',version="v1.0-mini",dataset_name='NuScenesDataset',max_sweeps=10)\
    #nuscenes_data_prep('/home/zp/data/nuscene-trainval',version="v1.0-trainval",dataset_name='NuScenesDataset',max_sweeps=10)\
    #nuscenes_data_prep('/home/zp/data/nuscene-test',version="v1.0-test",dataset_name='NuScenesDatasetVelo',max_sweeps=10)