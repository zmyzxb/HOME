import numpy as np
from torch.utils.data import Dataset,DataLoader
from scipy import sparse
import os
import copy
from tqdm import tqdm
from argoverse.data_loading.argoverse_forecasting_loader import ArgoverseForecastingLoader
from argoverse.map_representation.map_api import ArgoverseMap
import json
import torch
from torch import Tensor

def contain_pt(pt, size):
    return pt[0] >= 0 and pt[0] < size \
            and pt[1] >= 0 and pt[1] < size


def process(df, am, config):
    #---------parameter
    l = config['pred_range']
    gap = config['image_pixel_size']
    img_sz = len(np.arange(-l,l,gap))
    H = 20

    #---------output
    img = np.zeros((45,img_sz,img_sz),dtype='float64')
    agent_scalar_history = np.zeros((img_sz,img_sz,45),dtype='float64')
    other_scalar_history = None

    timestamp = np.sort(np.unique(df['TIMESTAMP'].values))
    agent_traj_full = df[df['OBJECT_TYPE']=='AGENT'][['X','Y']].values
    

    center = agent_traj_full[H-1,:]

    #image traj
    for i, ts in enumerate(timestamp):
        if i >= H:
            break
        df_i = df[df['TIMESTAMP']==ts]
        for _, row in df_i.iterrows():
            position = row[['X','Y']].values
            obj_type = row['OBJECT_TYPE']
            img_position = np.rint(((position - center + l) / gap).astype('float64')) \
            .astype('int').tolist()
            
            if img_position[0]>=0 and img_position[0]<img_sz \
                and img_position[1]>=0 and img_position[1]<img_sz:
                channel = i if obj_type == 'AGENT' else i + H
                img[channel][tuple(img_position)] += 1

    #image lanegraph
    lane_ids = am.get_lane_ids_in_xy_bbox(center[0], center[1], 'PIT', 50)
    polygons = am.find_local_driveable_areas([center[0]-l,center[0]+l,center[1]-l,center[1]+l],'PIT')

    img_driveable_area = np.zeros((img_sz,img_sz),dtype='float64')
    img_lane_boundaries = np.zeros((img_sz,img_sz),dtype='float64')
    img_centerline = np.zeros((3,img_sz,img_sz),dtype='float64')

    for i in polygons:
        for point in i:
            img_position = np.rint(((point[0:2] - center + l) / gap).astype('float64')) \
            .astype('int').tolist()
            #img_driveable_area[] += 1
            if contain_pt(img_position, img_sz):
                img_driveable_area[tuple(img_position)] = 1
    #centerline and lane_boundaries
    for lane_id in lane_ids:
        lane = am.city_lane_centerlines_dict['PIT'][lane_id]
        #print(lane.centerline)
        centerline = lane.centerline
        vector = centerline[-1] - centerline[0]
        line_mid = (centerline[-1] + centerline[0]) / 2
        
        polygon = am.get_lane_segment_polygon(lane_id, 'PIT')
        for pt in polygon:
            img_position = np.rint(((pt[0:2] - center + l) / gap).astype('float64')) \
            .astype('int').tolist()
            if contain_pt(img_position, img_sz):
                img_lane_boundaries[tuple(img_position)] = 255
        
        line_mid_img_position = np.rint(((line_mid[0:2] - center + l) / gap).astype('float64')) \
            .astype('int').tolist()

        turn_direction = 0
        if lane.turn_direction == 'LEFT':
            turn_direction = 0.5
        elif lane.turn_direction == 'RIGHT':
            turn_direction = 1
        if contain_pt(line_mid_img_position, img_sz):
            img_centerline[0,tuple(line_mid_img_position)] = turn_direction
            img_centerline[1:3,line_mid_img_position[0],line_mid_img_position[1]] \
                = vector / np.linalg.norm(vector)
    
    
    #calculate num and index of OTHERS vehicle
    car_id_map = dict()
    for _, row in df.iterrows():
        if row['OBJECT_TYPE'] != 'OTHERS':
            continue
        track_id = row['TRACK_ID'][-12:]
        vehicle_id = car_id_map[track_id] \
                if track_id in car_id_map else len(car_id_map)
        car_id_map[track_id] = vehicle_id

    num_of_others = len(car_id_map)
    #calculate scalar_history
    agent_scalar_history = np.zeros((3,H),dtype='float64')
    other_scalar_history = np.zeros((99,3,H),dtype='float64')
    #print(len(car_id_map))
    #print(car_id_map.keys())
    
    for i, ts in enumerate(timestamp):
        if i >= H:
            break
        df_i = df[df['TIMESTAMP']==ts]
        #print(agent_scalar_history[i,0:1].shape)
        agent_scalar_history[0:2,i] = (agent_traj_full[i,:] - center) / l #normalize
        for _, row in df_i.iterrows():
            if row['OBJECT_TYPE'] != 'OTHERS':
                continue
            position = (row[['X','Y']].values - center) / l #normalize
            track_id = row['TRACK_ID'][-12:]
            vehicle_id = car_id_map[track_id]
            if vehicle_id < 99:
                other_scalar_history[vehicle_id,0:2,i] = position

        for j in range(min(99,num_of_others)):
            if other_scalar_history[j,0,i] == 0:
                other_scalar_history[j,2,i] = 1
    
    img[40] = img_driveable_area
    img[41] = img_lane_boundaries
    img[42:45] = img_centerline
    
    agent_scalar_history = agent_scalar_history.reshape(1,3,H)
    scalar_history = np.concatenate((agent_scalar_history,other_scalar_history),axis=0)
    

    data_ = dict()
    data_['img'] = img
    data_['history'] = scalar_history
    data_['num_of_agents'] = min(100, num_of_others + 1)
    #label (traj groundtruth)
    final_pos = agent_traj_full[-1]
    label = np.rint(((final_pos - center + config['output_range']) / gap).astype('float64')) \
            .astype('int')
    label[0] = min(label[0], int(config['output_range']*2//gap))
    label[0] = max(label[0], 0)
    label[1] = min(label[1], int(config['output_range']*2//gap))
    label[1] = max(label[1], 0)
    #label = 
    #print(label)
    return data_,label

def collate(batch):
    device = torch.device('cuda')
    dt = dict()
    dt['img'] = torch.stack([Tensor(x['img']) for (x,y) in batch]).to(device)
    dt['history'] = torch.stack([Tensor(x['history']) for (x,y) in batch]).to(device)
    dt['num_of_agents'] = [x['num_of_agents'] for (x,y) in batch]
    label = np.stack([y for (x,y) in batch])
    return dt, label

class ArgoDataset(Dataset):
    """Some Information about MyDataset"""
    
    def __init__(self):
        super(ArgoDataset, self).__init__()
        config_file = open('config.json')
        self.config = json.load(config_file)
        split='train'
        self.avl = ArgoverseForecastingLoader(self.config['data_dir'][split])
        self.avl.seq_list = sorted(self.avl.seq_list)
        self.am = ArgoverseMap()
    def __getitem__(self, index):
        return process(copy.deepcopy(self.avl[index].seq_df), self.am, self.config)

    def __len__(self):
        return len(self.avl.seq_list)

if __name__ == '__main__':
    '''
    config_file = open('config.json')
    config = json.load(config_file)
    split='train'
    avl = ArgoverseForecastingLoader(config['data_dir'][split])
    avl.seq_list = sorted(avl.seq_list)
    am = ArgoverseMap()

    print('Processing {} data'.format(split))
    print('Num of data: {}'.format(len(avl)))

    for i in tqdm(range(len(avl))):
        #print(i)
        process(copy.deepcopy(avl[i].seq_df), am, config) 
        #break
    '''
    dataset = ArgoDataset()
    dl = DataLoader(dataset,batch_size = 2, collate_fn = collate)
    for idx, (dt, label) in enumerate(dl):
        print(label.shape)
        print(dt['img'].shape)
        print(dt['history'].shape)
        print(dt['num_of_agents'])
    #a = torch.randn((3,5))
    #b = a
    #c = [a,b]
    #print(torch.stack(c).shape)