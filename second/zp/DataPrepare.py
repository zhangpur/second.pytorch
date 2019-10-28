from nuscenes.nuscenes import NuScenes
import pandas as pd
import pickle
import numpy as np
import time
from pathlib import Path
import os

from second.zp.config import Configuration
from pyquaternion import Quaternion
from spconv.utils import VoxelGeneratorV2
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset


class DatasetWrapper(Dataset):
    """ convert our dataset to Dataset class in pytorch.
    """

    def __init__(self, dataset):
        self._dataset = dataset

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, idx):
        return self._dataset[idx]

    @property
    def dataset(self):
        return self._dataset

class NuScenesPrepare():
    NameMapping = {
        'movable_object.barrier': 'barrier',
        'movable_object.trafficcone': 'barrier',
        'vehicle.bicycle': 'cyclist',
        'vehicle.motorcycle': 'cyclist',
        'vehicle.bus.bendy': 'vehicle',
        'vehicle.bus.rigid': 'vehicle',
        'vehicle.car': 'vehicle',
        'vehicle.trailer': 'vehicle',
        'vehicle.truck': 'vehicle',
        'vehicle.construction': 'vehicle',
        'human.pedestrian.adult': 'pedestrian',
        'human.pedestrian.child': 'pedestrian',
        'human.pedestrian.construction_worker': 'pedestrian',
        'human.pedestrian.police_officer': 'pedestrian',
    }
    labelmapping=LabelEncoder()
    labelmapping.fit(['barrier','cyclist','pedestrian','vehicle'])
    DefaultAttribute = {
        "car": "vehicle.parked",
        "pedestrian": "pedestrian.moving",
        "trailer": "vehicle.parked",
        "truck": "vehicle.parked",
        "bus": "vehicle.parked",
        "motorcycle": "cycle.without_rider",
        "construction_vehicle": "vehicle.parked",
        "bicycle": "cycle.without_rider",
        "barrier": "",
        "traffic_cone": "",
    }

    def __init__(self,args):
        self.args_dp=args['DataPrepare']
        self.args_vg=args['VoxelGenerator']
        self.cache_path=self.args_dp.data_root+'/'+self.args_dp.cache_name

        if os.path.exists(self.cache_path):
            self._Data_frags=pickle.load(open(self.cache_path, 'rb'))
        else:
            self.nusc = NuScenes(version=self.args_dp.version, dataroot=self.args_dp.data_root, verbose=self.args_dp.verbose)
            self._Data_frags=self.getFragAnnotations()
            pickle.dump(self._Data_frags,open(self.cache_path, 'wb'))

        if True:
            self._Data_frags=[item for scene_data in self._Data_frags for item in scene_data]

    def __len__(self):
        return len(self._Data_frags)

    def __getitem__(self, idx):
        c_frag = self._Data_frags[idx]
        res = {
            "lidar": {
                "type": "lidar",
                "points": None,
            },
            "gt_boxes": {
                "obs": None,
                "pred": None,
            },
            "metadata": {
                "token": c_frag['Data_frags'][self.args_dp.obs_length-1] # token of last observed frame
            },
        }
        sweep_voxels,sweep_coords,sweep_num_voxels,sweep_num_points=[],[],[],[]
        bev_imgs, cam_imgs = [], []
        gt_boxes=[]
        #ts = c_frag['Data_frags'][self.args.obs_length-1]["timestamp"] / 1e6

        #get BEV sweeps
        for fi,c_frame in enumerate(c_frag['Data_frags']):
            if fi < self.args_dp.obs_length:
                # get Annotations
                gt_boxes.append(c_frame['boxes'])
                # load lidar points
                lidar_path = c_frame['lidar_path']
                points = np.fromfile(
                    str(Path(self.args_dp.data_root)/lidar_path), dtype=np.float32, count=-1).reshape([-1, 5])
                points[:, 3] /= 255
                points[:, :3] = points[:, :3] @ c_frame["R_cl2rl"].T
                points[:, :3] += c_frame["T_cl2rl"]
                # generate voxel bev
                voxel_res=self.getVoxel(points)
                if 'bev_img' in self.args_vg.bev_data:
                    bev_img_size = [np.int(np.ceil((self.args_vg.point_cloud_range[3 + i] - self.args_vg.point_cloud_range[i])
                                       / self.args_vg.voxel_size[i])) for i in range(3)]
                    bev_img = np.zeros(bev_img_size)
                    bev_img[voxel_res['coordinates'][:, 2], voxel_res['coordinates'][:, 1], voxel_res['coordinates'][:, 0]]\
                    =voxel_res["num_points_per_voxel"]
                    bev_imgs.append(bev_img)
                if 'bev_index' in self.args_vg.bev_data:
                    sweep_voxels.append(voxel_res['voxels'])
                    sweep_coords.append(voxel_res['coordinates'])
                    sweep_num_voxels.append(np.array([voxel_res['voxels'].shape[0]], dtype=np.int64))
                    sweep_num_points.append( voxel_res["num_points_per_voxel"])
                # Load image
                if self.args_dp.use_image == 'last_image': # only use image of the last observed frame
                    load_image = fi == self.args_dp.obs_length - 1
                elif self.args_dp.use_image == 'key_images': # use image of all key frames
                    load_image = 'cam_path' in c_frame.keys()
                if load_image:
                    if Path(c_frame['cam_path']).exists():
                        with open(str(c_frame['cam_path']), 'rb') as f:
                            image_str = f.read()
                    else:
                        image_str=None
                    cam_imgs.append(image_str)

        res["lidar"]["voxel_sweeps"] =  np.concatenate(sweep_voxels, axis=0)
        res["lidar"]["bev_imgs"] =  np.stack(bev_imgs, axis=0)
        res["lidar"]["coordinates"] =  np.concatenate(sweep_coords, axis=0)
        res["lidar"]["num_voxels"] =  np.concatenate(sweep_num_voxels, axis=0)
        res["lidar"]["num_points"] =  np.concatenate(sweep_num_points, axis=0)
        res["cam"] = {
            "type": "camera",
            "data": image_str,
            "datatype": Path(c_frag['Data_frags'][self.args_dp.obs_length-1]['cam_path']).suffix[1:],
        }
        gt_boxes=np.stack(gt_boxes,axis=0)
        res["gt_boxes"]["obs"]=gt_boxes[:self.args_dp.obs_length]
        res["gt_boxes"]["pred"] =gt_boxes[self.args_dp.obs_length:]
        res['cls_label']=self.labelmapping.fit_transform(c_frag['names'])

        return res
        #Ground Truth (by instance)
        #res["GT"]["obs"]=
        #image of last
    def getVoxel(self,points):
        max_voxels=100000
        voxel_generator = VoxelGeneratorV2(
            voxel_size=list(self.args_vg.voxel_size),
            point_cloud_range=list(self.args_vg.point_cloud_range),
            max_num_points=self.args_vg.max_number_of_points_per_voxel,
            max_voxels=max_voxels,
            full_mean=self.args_vg.full_empty_part_with_mean,
            block_filtering=self.args_vg.block_filtering,
            block_factor=self.args_vg.block_factor,
            block_size=self.args_vg.block_size,
            height_threshold=self.args_vg.height_threshold)

        res = voxel_generator.generate(
            points, max_voxels)
        return res
    def getSamplebyFrame(self):
        #################interplot data 10hz
        scene_all=[]
        for si,scene in enumerate(self.nusc.scene):
            sample_interp_all = []
            first_sample = self.nusc.get('sample', scene['first_sample_token'])
            sd_rec = self.nusc.get('sample_data', first_sample['data']["LIDAR_TOP"])
            sample_interp_all.append(sd_rec)
            while sd_rec['next'] != '':
                sd_rec = self.nusc.get('sample_data', sd_rec['next'])
                sample_interp_all.append(sd_rec)
            scene_all.append(sample_interp_all)
        return scene_all
    def getFragAnnotations(self):

        scene_frames = self.getSamplebyFrame()
        Data_frags=[]
        key_slide_window=int(self.args_dp.obs_length*2) #find key frame in this time window
        si_start=0
        if os.path.exists(self.cache_path):
            Data_frags=pickle.load(open(self.cache_path, 'rb'))
            si_start=len(Data_frags)
        print('-------------Prepraing fragments--------------')
        for si in range(si_start,len(scene_frames)):
            scene_data=scene_frames[si]
            start = time.time()
            scene_frags = []
            for di,sample_data in enumerate(scene_data):
                frag_info={}
                if sample_data['is_key_frame']:
                    if di <= self.args_dp.obs_length or di >= len(scene_data)-self.args_dp.pred_length*self.args_dp.interval:
                        continue
                    cur_frag_index=[i+1 for i in range(di-self.args_dp.obs_length,di+self.args_dp.pred_length)]#the fragment index
                    if di !=cur_frag_index[self.args_dp.obs_length-1]:
                        print('error')
                    start_key=max(0,min(di-self.args_dp.obs_length,di - key_slide_window))
                    end_key=min(len(scene_data)-1,max(di+self.args_dp.pred_length,di + key_slide_window))
                    cur_key_index = [i+1 for i in range(start_key,end_key)]#find key frame in this index

                    ## Get reference coordinates
                    refer_frame = sample_data
                    refer_cs_rec = self.nusc.get('calibrated_sensor', refer_frame['calibrated_sensor_token'])
                    refer_pos_rec = self.nusc.get('ego_pose', refer_frame['ego_pose_token'])
                    R_rl2re, T_rl2re = refer_cs_rec['rotation'], refer_cs_rec['translation']
                    R_re2g, T_re2g = refer_pos_rec['rotation'], refer_pos_rec['translation']
                    R_rl2re_mat = Quaternion(R_rl2re).rotation_matrix
                    R_re2g_mat = Quaternion(R_re2g).rotation_matrix

                    # get key frame location
                    key_frame_flag = np.zeros((len(cur_key_index),), dtype='bool')
                    key_frame_index = []
                    for i, d in enumerate(cur_key_index):
                        try:
                            if scene_data[d]['is_key_frame'] == True:
                                key_frame_index.append(i)
                                key_frame_flag[i] = True
                        except:
                            print('error')
                    key_frames = np.array(scene_data[cur_key_index[0]:(cur_key_index[-1]+1)])[key_frame_flag]

                    # only key frame has annotations, so firstly get key frame infos
                    key_sample, key_sample_token, key_instances, key_annotations, key_velocity = [], [], [], [], []
                    for k, key_frame in enumerate(key_frames):
                        sample_token = key_frame['sample_token']
                        sample = self.nusc.get('sample', sample_token)
                        annotations = [
                            self.nusc.get('sample_annotation', token)
                            for token in sample['anns']
                        ]
                        velocity = np.array(
                            [self.nusc.box_velocity(token)[:2] for token in sample['anns']])
                        key_sample_token.append(sample_token)
                        key_instances.append([anno['instance_token'] for anno in annotations])
                        key_sample.append(sample)
                        key_annotations.append(annotations)
                        key_velocity.append(velocity)

                    # get full presented instance token in the candidate fragments
                    instances_intersect = list(set.intersection(*[set(i) for i in key_instances]))
                    #instances_union = list(set.union(*[set(i) for i in key_instances]))

                    # full presented instance flags
                    valid_inst_flags = [np.zeros(len(kinst), dtype='bool') for kinst in key_instances]
                    for kinst, key_inst in enumerate(key_instances):
                        for vkinst, valid_inst in enumerate(key_inst):
                            if valid_inst in instances_intersect:
                                valid_inst_flags[kinst][vkinst] = True

                    ##########################################
                    ##             Prepare fragments Database
                    ##########################################
                    cur_key_frame_index = []
                    for i, d in enumerate(cur_frag_index):
                        if scene_data[d]['is_key_frame'] == True:
                            cur_key_frame_index.append(i)

                    frag_info['Data_frags'] = []
                    frag_info['instance_token'] = instances_intersect
                    frag_info['key_frame_index'] = cur_key_frame_index
                    frag_info['last_obs_frame']=di
                    frag_info['scene_No'] = si
                    for i, d in enumerate(cur_frag_index):

                        frag_data = {}
                        sample_data=scene_data[d]
                        sample_token = sample_data['sample_token']
                        sample = self.nusc.get('sample', sample_token)

                        # find the key sample this frame data belongs to
                        try:
                            key_sample_ind = key_sample_token.index(sample_token)
                        except:
                            print('can not find corresponding  key frame at scene {} frame {}'.format(si,d))

                        valid_inst_flag = valid_inst_flags[key_sample_ind]

                        ## Pose matrix: lidar2ego and ego2global
                        s_cs_rec = self.nusc.get('calibrated_sensor', sample_data['calibrated_sensor_token'])
                        s_pos_rec = self.nusc.get('ego_pose', sample_data['ego_pose_token'])

                        R_cl2ce, T_cl2ce = s_cs_rec['rotation'], s_cs_rec['translation']
                        R_ce2g, T_ce2g = s_pos_rec['rotation'], s_pos_rec['translation']

                        R_cl2ce_mat = Quaternion(R_cl2ce).rotation_matrix
                        R_ce2g_mat = Quaternion(R_ce2g).rotation_matrix

                        # Data_frag['Info_frags']['T_l2e'],Data_frag['Info_frags']['R_l2e'] = cs_record['translation'],cs_record['rotation']
                        # Data_frag['Info_frags']['T_e2g'],Data_frag['Info_frags']['R_e2g'] = pose_record['translation'], pose_record['rotation']

                        ## Get Relative Pose: R_cl2rl,T_cl2rl, based on R/T_rl2re, R/T_re2g, R/T_cl2ce, R/T_ce2g
                        # r: reference, c: current, l: lidar, e: ego, g: global
                        # Attention: R_b2a = inv(R_a2b), T_b2a = - T_b2a * inv(R_b2a),

                        # R_cl2rl = R_cl2se * R_ce2g * [R_g2rl]
                        # R_g2rl= R_g2re * R_re2rl =  inv(R_re2g) * inv(R_rl2re)
                        R_cl2rl = (R_cl2ce_mat.T @ R_ce2g_mat.T) @ (
                                np.linalg.inv(R_re2g_mat).T @ np.linalg.inv(R_rl2re_mat).T)

                        # T_cl2rl = (T_cl2ce * R_ce2g + T_ce2g) * [R_g2rl] + [T_g2rl]
                        # T_g2rl = (T_g2re * R_re2rl + T_re2rl) = - T_re2g * inv(R_re2g) - T_rl2re * inv(R_rl2re)
                        T_cl2rl = (T_cl2ce @ R_ce2g_mat.T + T_ce2g) @ (
                                    np.linalg.inv(R_re2g_mat).T @ np.linalg.inv(R_rl2re_mat).T) \
                                  - T_re2g @ (np.linalg.inv(R_re2g_mat).T @ np.linalg.inv(
                            R_rl2re_mat).T) - T_rl2re @ np.linalg.inv(R_rl2re_mat).T

                        frag_data['R_cl2rl'], frag_data['T_cl2rl'] = R_cl2rl, T_cl2rl

                        ### Get valid boxes.Then Transform to the reference coordinates
                        boxes = self.nusc.get_boxes(sample_data['token'])  # At global coordinate
                        for box in boxes:
                            # Move box to referred coord system
                            box.translate(-np.array(refer_pos_rec['translation']))
                            box.rotate(Quaternion(refer_pos_rec['rotation']).inverse)
                            box.translate(-np.array(refer_cs_rec['translation']))
                            box.rotate(Quaternion(refer_cs_rec['rotation']).inverse)

                        boxes = np.array(boxes)  # At reference coordinate
                        try:
                            valid_boxes = boxes[valid_inst_flag]
                        except:
                            print('can not find valid box at scene {} frame {}'.format(si, d))
                        ## Transform Boxes to [location,dimension,rotation]
                        locs = np.array([b.center for b in valid_boxes]).reshape(-1, 3)
                        dims = np.array([b.wlh for b in valid_boxes]).reshape(-1, 3)
                        rots = np.array([b.orientation.yaw_pitch_roll[0]
                                         for b in valid_boxes]).reshape(-1, 1)
                        gt_boxes = np.concatenate([locs, dims, -rots - np.pi / 2], axis=1)
                        frag_data['boxes'] = gt_boxes

                        ## Datapath
                        if i<self.args_dp.obs_length:
                            frag_data['lidar_path'] = sample_data['filename']
                            if sample_data['is_key_frame']:
                                cam_front_token = sample["data"]["CAM_FRONT"]
                                cam_path, _, _ = self.nusc.get_sample_data(cam_front_token)
                                frag_data['cam_path'] = cam_path

                        ## Object name
                        if 'names' not in frag_info.keys():
                            names = [b.name for b in valid_boxes]
                            for i in range(len(names)):
                                if names[i] in self.NameMapping:
                                    names[i] = self.NameMapping[names[i]]
                            names = np.array(names)
                            frag_info['names'] = names

                        ##Velocity (without interplotion)
                        valid_velo = key_velocity[key_sample_ind][valid_inst_flag]
                        # convert velo from global to current lidar
                        for i in range(len(valid_boxes)):
                            velo = np.array([*valid_velo[i], 0.0])
                            velo = velo @ np.linalg.inv(R_ce2g_mat).T @ np.linalg.inv(R_cl2ce_mat).T
                            valid_velo[i] = velo[:2]
                        frag_data['Velocity'] = valid_velo
                        frag_data['FrameNo.'] = d
                        frag_data['Token'] = sample_data['token']
                        frag_data['timestamp'] = sample_data['timestamp']
                        frag_info['Data_frags'].append(frag_data)
                    scene_frags.append(frag_info)
            Data_frags.append(scene_frags)
            end = time.time()
            print('scene {}/{}: total frags: {} time: {} '.format(si, len(scene_frames), len(scene_frags), end - start))
            if si%200==0 and si>0:
                pickle.dump(Data_frags, open(self.cache_path, 'wb'))
        return Data_frags
if __name__ == '__main__':

    args = Configuration('DataPrepare','VoxelGenerator')

    NSUparepare=NuScenesPrepare(args)




