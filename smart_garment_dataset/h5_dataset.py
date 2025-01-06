import h5py
import torch
import cv2
import numpy as np

from kinematics import INVERSE_KINEMATICS
from config import OPENPOSE_15_CONFIG

'''
h5_dataset keys:
'device': '2' represent the smart garment, no use in this project
'gender': '0' represents male and '1' represents female
'joint': [N, 15, 3] 3D pose label
'person': participant identifier
'posture': [N, ] pose identifier for classification
'pressure': [N, 2, 64, 32] pressure data
'''

class H5_DATASET:
    '''
    The class is designed to access .h5 dataset.
    '''
    def __init__(self, participant_id: int, section_id: int):

        self.config = OPENPOSE_15_CONFIG()
        self.inverse_k_unit = INVERSE_KINEMATICS(self.config)
        self.dataset = h5py.File(f'./data_sample/participant{participant_id}_section{section_id}.h5', 'r')
        '''
        Slices [0, 2, 1] is used to transfer y and z axis, after which z axis represent the actual vertical direction
        This procedure makes the following view changing easier.
        '''
        self.pose_3d = self.dataset['joint'][:][:, :, [0, 2, 1]]
        self.pressure = self.dataset['pressure'][:]
        self.template = np.load(f'./data_sample/template/template_{participant_id}.npy')[:, [0, 2, 1]]

    def turn_pose_egocentric(self, pose_3d: np.ndarray):
        '''
        This function turns 3d poses from camera's global views to participant's egocentric views.
        The view is transformed by removing the rotation component along vertical direction (z axis).
        '''
        R_global, R_local = self.inverse_k_unit.inverse_kinematics(pose_3d, self.template)
        R_z, R_y, R_x = self.inverse_k_unit.decomposite_rotation_matrix(R_global[0])

        return np.dot(np.linalg.inv(R_z), pose_3d.T).T
    
    def pressure_image_process(self):
        '''
        Process all pressure images in the dataset. The function limits pressure values and smoothes the images.
        '''
        for i in range(self.pressure.shape[0]):
            img_front = self.pressure[i,0]
            img_back = self.pressure[i,1]
            '''
            threshold 1024 is used to cut values caused by short-circuits.
            threshold 512 is used to limit pressure values, making it easier for normalization.
            '''
            ret, img_front = cv2.threshold(img_front, 1024, 1024, cv2.THRESH_TOZERO_INV)
            ret, img_front = cv2.threshold(img_front, 512, 512, cv2.THRESH_TRUNC)
            img_front = cv2.GaussianBlur(img_front, (3, 3), 0, 0)

            ret, img_back = cv2.threshold(img_back, 1024, 1024, cv2.THRESH_TOZERO_INV)
            ret, img_back = cv2.threshold(img_back, 512, 512, cv2.THRESH_TRUNC)
            '''
            Gaussian filer is used to recover 0-values caused by broken-circuits.
            '''
            img_back = cv2.GaussianBlur(img_back, (3, 3), 0, 0)
            
            self.pressure[i] = np.array([img_front, img_back])[:]
        return
    
    def select_data_by_pose(self, pose_id:int):
        '''
        Select pressure data and pose labels of a certian pose.
        Inputing pose_id = -1 will return all poses.
        '''
        posture_value = self.dataset['posture'][:]

        if(pose_id == -1):
            return (self.pressure[:], self.pose_3d[:])
        
        index = np.where(posture_value == pose_id)[0]
        return (self.pressure[index][:], self.pose_3d[index][:])
    
    def make_tensor(self, window_len:int, pose_id:int):
        '''
        Package tensor dataset of a certain pose for model training and validation.
        The pressure data will be normalized to [0, 1] by being divided by max-value 512.
        '''
        data_npy = []
        label_npy = []

        data, label = self.select_data_by_pose(pose_id)
        index = 0
        while(index + window_len < data.shape[0]):
            data_npy.append(data[index: index + window_len])
            label_npy.append(np.concatenate([self.turn_pose_egocentric(label[index + window_len//2]).flatten(), 
                                             self.template.flatten()]))
            index = index + 1
        
        data_npy = np.array(data_npy) / 512
        label_npy = np.array(label_npy)

        data_tensor = torch.tensor(data_npy).float()
        label_tensor = torch.tensor(label_npy).float()

        return (data_tensor, label_tensor)
