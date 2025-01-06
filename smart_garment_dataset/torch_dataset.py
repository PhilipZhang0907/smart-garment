import torch
from torch.utils.data import Dataset
from h5_dataset import H5_DATASET

class SMART_GARMENT_DATASET(Dataset):
    '''
    Torch dataset directly accessed by the neural network.
    '''
    def __init__(self, grouping:tuple, window_len:int, pose_id:int):
        '''
        grouping is a list consists of (name, section) pairs
        '''
        self.data_tensor, self.label_tensor = self.make_tensor(grouping, window_len, pose_id)
        print(f'Dataset data size: {self.data_tensor.size()}')
        print(f'Dataset label size: {self.label_tensor.size()}')

    def make_tensor(self, grouping:list, window_len:int, pose_id:int):

        data_tensor = []
        label_tensor = []

        for (name, section) in grouping:

            print(f'name: {name}, section: {section}.')

            h5_dataset = H5_DATASET(name, section)
            h5_dataset.pressure_image_process()

            data, label = h5_dataset.make_tensor(window_len, pose_id)

            data_tensor.append(data)
            label_tensor.append(label)
        
        data_tensor = torch.cat(data_tensor, dim=0)
        label_tensor = torch.cat(label_tensor, dim=0)

        return (data_tensor, label_tensor)

    def __getitem__(self, index):
        return self.data_tensor[index], self.label_tensor[index]
    
    def __len__(self):
        return self.data_tensor.size()[0]    
        