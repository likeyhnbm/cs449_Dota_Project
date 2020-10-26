import json
import pandas as pd
import glob
import seaborn as sns
import torch
import cv2

from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt

class DotaTrainDataset(Dataset):


    def __init__(self, list_name='train_split.txt',json_dir="annotations",frames_dir="frames",frame_size=32):

        # 
        # List_name : a txt file containing the list of video file.
        # json_dir : the folder name of json files.
        # frames_dir: the folder name of fames.
        # frame_size: the target length of x.
        # 
        
        self.frames_dir = frames_dir
        self.frame_size = frame_size
        #read file list
        f=open(list_name)
        file_list = f.read().splitlines()

        #Check if these files exist
        exists = glob.glob(frames_dir+"/*")

        exists = list(map(lambda x: x.replace(frames_dir+"\\",''),exists))

        broken_list = []

        for i,x in enumerate(file_list):
            if x not in exists:
                broken_list.append(x)

        for item in broken_list:
            file_list.remove(item)

        self.jsons = []

        for filename in file_list:
            info = pd.read_json(json_dir+'/'+filename+".json")
            self.jsons.append(info)


    def __len__(self):
        return 2*len(self.jsons)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        label = idx%2
        
        info = self.jsons[idx//2]
        file_name = info['video_name'][0]
        anomaly_start = info['anomaly_start'][0]
        anomaly_end = info['anomaly_end'][0]
        anomaly_length = anomaly_end - anomaly_start
        
        #For normal
        if not label:
            imgs = []
            print(1)
            for i in range(anomaly_start):
                img_name = self.frames_dir+"/"+file_name+"/%06d.jpg" % i 
               
                img = cv2.imread(img_name)
               
                imgs.append(img)
            xs = []
            
            for i in range(self.frame_size):
                xs.append(imgs[i*anomaly_length // self.frame_size])
        #For normal
        else:
            imgs = []
            for i in range(anomaly_start,anomaly_end):
                img_name = self.frames_dir+"/"+file_name+"/%06d.jpg" % i 
                img = cv2.imread(img_name)
                imgs.append(img)
            xs = []
            
            for i in range(self.frame_size):
                xs.append(imgs[i*anomaly_length // self.frame_size])
        
        
        
        return np.array(xs),label