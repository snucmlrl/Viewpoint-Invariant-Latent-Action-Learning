import os

from glob import glob
import cv2

from .base_dataset import BaseDataset

class XSkillDataset(BaseDataset):
    def __init__(
        self,
        data_path: str='/workspace/datasets/xskill',
        unseen_type: str='none',
        **kwargs,
    ):
        # assert and error message if unseen_type is not in the list
        assert unseen_type in ['none', 'human', 'robot', 'task'], \
            f"unseen_type must be one of ['none', 'human', 'robot'], but got {unseen_type}"
        self.unseen_type = unseen_type
        
        kwargs['min_predict_future_horizon'] = 10
        kwargs['max_predict_future_horizon'] = 20
        super().__init__(data_path, **kwargs)
        
    def _prepare_data(self, data_path):
        
        human_videos = []
        robot_videos = []
        for task in os.listdir(os.path.join(data_path, 'data_v2')):
            task_path = os.path.join(data_path, 'data_v2', task, 'videos')
            demo_videos = []
            for demo in sorted(os.listdir(task_path)):
                videos = glob(os.path.join(task_path, demo, '[!1]', '*.mp4'))
                demo_videos.extend(videos)
            demo_num = len(demo_videos)
            if self.unseen_type == 'none':
                if self.train:
                    demo_videos = demo_videos[:int(demo_num*0.9)]
                else:
                    demo_videos = demo_videos[int(demo_num*0.9):]
            if task.startswith('human'):
                human_videos.extend(demo_videos)
            else:
                robot_videos.extend(demo_videos)
                
        if self.unseen_type == 'human':
            if self.train:
                videos = robot_videos
            else:
                videos = human_videos
        elif self.unseen_type == 'robot':
            if self.train:
                videos = human_videos
            else:
                videos = robot_videos
        else:
            videos = human_videos + robot_videos

        image_pair = []

        for video in videos:
            vid_len = cv2.VideoCapture(video).get(cv2.CAP_PROP_FRAME_COUNT)
            if vid_len < self.min_predict_future_horizon:
                continue

            image_pair.append({
                'path': video,
                'length': vid_len,
            })
            
        self.image_pair = image_pair