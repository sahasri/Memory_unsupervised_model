import numpy as np
from collections import OrderedDict
import os
import glob
import cv2
import torch.utils.data as data

class DataLoader(data.Dataset):
    def __init__(self, video_folder, transform, resize_height, resize_width, time_step=4, num_pred=1):
        self.dir = video_folder
        self.transform = transform
        self.videos = OrderedDict()
        self._resize_height = resize_height
        self._resize_width = resize_width
        self._time_step = time_step
        self._num_pred = num_pred
        self.data = self.setup()
        self.samples = self.get_all_samples()
        
    def setup(self):
        videos = glob.glob(os.path.join(self.dir, '*'))
        for video in sorted(videos):
            video_name = video.split('/')[-1]
            self.videos[video_name] = {}
            self.videos[video_name]['path'] = video
            self.videos[video_name]['length'] = self.get_video_length(video)
        
    def get_video_length(self, video_path):
        cap = cv2.VideoCapture(video_path)
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return length
            
    def get_all_samples(self):
        frames = []
        for video_name, info in self.videos.items():
            video_path = info['path']
            cap = cv2.VideoCapture(video_path)
            for i in range(info['length'] - self._time_step):
                frames.append((video_name, i))
            cap.release()
        return frames               
            
    def __getitem__(self, index):
        video_name, frame_index = self.samples[index]
        batch = []
        cap = cv2.VideoCapture(self.videos[video_name]['path'])
        for i in range(frame_index, frame_index + self._time_step + self._num_pred):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                image = cv2.resize(frame, (self._resize_width, self._resize_height))
                image = image.astype(dtype=np.float32)
                image = (image / 127.5) - 1.0
                if self.transform is not None:
                    batch.append(self.transform(image))
            else:
                raise ValueError("Failed to read frame from video.")
        cap.release()
        return np.concatenate(batch, axis=0)
        
    def __len__(self):
        return len(self.samples)
