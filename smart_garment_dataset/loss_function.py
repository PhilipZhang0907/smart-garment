import torch
from torch import nn
from kinematics import FORWARD_KINEMATICS
from config import OPENPOSE_15_CONFIG

class L2_LOSS(nn.Module):
    
    def __init__(self):
        super(L2_LOSS, self).__init__()
        self.forward_k_unit = FORWARD_KINEMATICS()

    def forward(self, pred: torch.Tensor, y: torch.Tensor):
        '''
        1. convert r6d output (15, 6) to rotation matrix (15, 3, 3)
        2. reconstruct joint location (15, 3, 3) using template
        3. calculate L2 loss
        '''
        r6d = pred.view(-1, 15, 6)
        y_location = y[:, 0:45].view(-1, 15, 3)
        y_template = y[:, 45:90].view(-1, 15, 3)

        pred_R_local = self.forward_k_unit.r6d_to_rotation_matrix(r6d)
        pred_R_global = self.forward_k_unit.forward_tree_batch(pred_R_local, None)
        pred_location = self.forward_k_unit.forward_kinematics_batch(pred_R_global, y_template).view(-1, 15, 3)
        
        L2_loss = torch.mean(torch.norm(pred_location - y_location, dim=2, p=2), dim=0)
        return L2_loss
    
class LC_LOSS(nn.Module):

    def __init__(self):
        super(LC_LOSS, self).__init__()
        self.config = OPENPOSE_15_CONFIG()
        self.forward_k_unit = FORWARD_KINEMATICS()

    def transfer_relative_location(self, x: torch.Tensor):
        parent_location = [x[:, 0, :]]
        for i in range(1, 15):
            parent_location.append(x[:, self.config.PARENT[i], :])
        parent_location = torch.stack(parent_location, dim=1)
        return x - parent_location

    def forward(self, pred: torch.Tensor, y: torch.Tensor):
        '''
        1. convert r6d output (15, 6) to rotation matrix (15, 3, 3)
        2. reconstruct joint location (15, 3, 3) using template
        3. calculate LC loss
        '''
        r6d = pred.view(-1, 15, 6)
        y_location = y[:, 0:45].view(-1, 15, 3)
        y_template = y[:, 45:90].view(-1, 15, 3)
        y_rotation = y[:, 90:99].view(-1, 1, 3, 3)

        pred_R_local = self.forward_k_unit.r6d_to_rotation_matrix(r6d)
        pred_R_global = self.forward_k_unit.forward_tree_batch(pred_R_local, y_rotation)
        pred_location = self.forward_k_unit.forward_kinematics_batch(pred_R_global, y_template).view(-1, 15, 3)

        relative_y = self.transfer_relative_location(y_location)
        relative_pred = self.transfer_relative_location(pred_location)

        lc_loss = 50.0 * torch.mean(1.0 - torch.nn.functional.cosine_similarity(relative_y, relative_pred, dim=2), dim=0)
        return lc_loss
