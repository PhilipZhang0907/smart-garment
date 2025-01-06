import torch
import cv2
import math
import numpy as np
from config import OPENPOSE_15_CONFIG

class INVERSE_KINEMATICS:

    def __init__(self):
        self.config = OPENPOSE_15_CONFIG()
    
    def rodrigues(self, vec_before: np.ndarray, vec_after: np.ndarray):
        
        if(np.allclose(vec_before, vec_after)):
            return np.identity(3)
        vec_axis = np.cross(vec_before, vec_after)
        rotate_angle = np.arccos(np.dot(vec_before, vec_after) / 
                                 (np.linalg.norm(vec_before) * np.linalg.norm(vec_after)))
        vec_axis = rotate_angle * (vec_axis / np.linalg.norm(vec_axis))
        rotation_mat, jacobian = cv2.Rodrigues(vec_axis)
        return rotation_mat
    
    def svd_root_rotation(self, joint_location: np.ndarray, template_location: np.ndarray):
        
        points_index = [self.config.LHIP, self.config.RHIP, self.config.NECK]
        p1 = np.array(joint_location[points_index])
        p2 = np.array(template_location[points_index])
        q1 = p1 - joint_location[self.config.ROOT]
        q2 = p2 - template_location[self.config.ROOT]
        H = np.matmul(q2[0].reshape((3, 1)), q1[0].reshape(1, 3)) + \
            np.matmul(q2[1].reshape((3, 1)), q1[1].reshape(1, 3)) + \
            np.matmul(q2[2].reshape((3, 1)), q1[2].reshape(1, 3))
        U, sigma, VT = np.linalg.svd(H)
        diag = np.array([[1, 0, 0],
                         [0, 1, 0],
                         [0, 0, np.linalg.det(np.matmul(VT.T, U.T))]])
        R = np.matmul(np.matmul(VT.T, diag), U.T)
        return R
    
    def decomposite_rotation_matrix(self, mat: np.ndarray):
        '''
        INPUT:     Rotation matrix 'mat'
        OUTPUT:    Components of 'mat' along x,y,z axis
        ATTENTION: Note the order 'mat = RzRyRx'
        '''
        theta_x = math.atan2(mat[2, 1], mat[2, 2])
        theta_y = math.atan2(-1*mat[2, 0], math.sqrt(mat[2, 1]**2 + mat[2, 2]**2))
        theta_z = math.atan2(mat[1, 0], mat[0, 0])

        R_x = np.array([[1, 0, 0],
                        [0, math.cos(theta_x), -1 * math.sin(theta_x)],
                        [0, math.sin(theta_x), math.cos(theta_x)]])

        R_y = np.array([[math.cos(theta_y), 0, math.sin(theta_y)],
                        [0, 1, 0],
                        [-1 * math.sin(theta_y), 0, math.cos(theta_y)]])
        
        R_z = np.array([[math.cos(theta_z), -1 * math.sin(theta_z), 0],
                        [math.sin(theta_z), math.cos(theta_z), 0],
                        [0, 0, 1]])
    
        return (R_z, R_y, R_x)

    def inverse_tree(self, joint_location: np.ndarray, template_location: np.ndarray, R_global: list, R_local: list):

        for i in range(1, len(self.config.PARENT)):
            pk = joint_location[i]
            pak = joint_location[self.config.PARENT[i]]
            tk = template_location[i]
            tpak = template_location[self.config.PARENT[i]]
            p = pk - pak
            t = tk - tpak
            p = np.dot(np.linalg.inv(R_global[self.config.PARENT[i]]), p.T)
            R_local.append(self.rodrigues(t, p))
            R_global.append(np.dot(R_global[self.config.PARENT[i]], self.rodrigues(t, p)))
        R_global = np.array(R_global)
        R_local = np.array(R_local)
        return (R_global, R_local)
    
    def inverse_kinematics(self, joint_location: np.ndarray, template_location: np.ndarray):
        
        root_rotation = self.svd_root_rotation(joint_location, template_location)

        R_global = [root_rotation]
        R_local = [root_rotation]

        R_global, R_local = self.inverse_tree(joint_location, template_location, R_global, R_local)
        return (R_global, R_local)

class FORWARD_KINEMATICS:

    def __init__(self):
        self.config = OPENPOSE_15_CONFIG()

    def normalize_tensor(self, x: torch.Tensor, dim=-1):
        
        norm = x.norm(dim=dim, keepdim=True)
        normalized_x = x / norm
        normalized_x[torch.isnan(normalized_x)] = 0
        return normalized_x
    
    def r6d_to_rotation_matrix(self, r6d: torch.Tensor):
        
        r6d = r6d.view((-1, 6))

        column0 = self.normalize_tensor(r6d[:, 0:3])
        column1 = self.normalize_tensor(r6d[:, 3:6] - (column0 * r6d[:, 3:6]).sum(dim=1, keepdim=True) * column0)
        column2 = column0.cross(column1, dim=1)
        
        r = torch.stack((column0, column1, column2), dim=-1)
        r[torch.isnan(r)] = 0
        
        return r.view((-1, self.config.NUM_JOINT, 3, 3))

    def forward_tree_batch(self, R_local: torch.Tensor, R_root):
        
        R_local = R_local.view(R_local.shape[0], -1, 3, 3)

        if R_root is None:
            R_global = [R_local[:, 0]]
        else:
            R_global = [torch.bmm(R_local[:, 0], R_root[:, 0])]
        
        for i in range(1, len(self.config.PARENT)):
            R_global.append(torch.bmm(R_global[self.config.PARENT[i]], R_local[:, i]))

        R_global = torch.stack(R_global, dim=1)
        return R_global
    
    def forward_tree(self, R_local: np.ndarray, R_root):

        R_local = R_local.reshape(-1, 3, 3)

        if R_root is None:
            R_global = [R_local[0]]
        else:
            R_global = [np.dot(R_local[0], R_root)]

        for i in range(1, len(self.config.PARENT)):
            R_global.append(np.dot(R_global[self.config.PARENT[i]], R_local[i]))

        return np.array(R_global)
    
    def forward_kinematics_batch(self, R_global: torch.Tensor, template_location: torch.Tensor):
        
        template_location = template_location.view(-1, self.config.NUM_JOINT, 3, 1)
        joint_location = [torch.bmm(R_global[:, 0], template_location[:, 0])]

        for i in range(1, len(self.config.PARENT)):
            joint_location.append(torch.bmm(R_global[:, i], 
                                            (template_location[:, i] - template_location[:, self.config.PARENT[i]]))
                                    + joint_location[self.config.PARENT[i]])
        joint_location = torch.stack(joint_location, dim=1)
        return joint_location
    
    def forward_kinematics(self, R_global: np.ndarray, template_location: np.ndarray):

        template_location = template_location.reshape(-1, 3, 1)
        joint_location = [np.dot(R_global[0], template_location[0])]

        for i in range(1, len(self.config.PARENT)):
            joint_location.append(np.dot(R_global[i],
                                  (template_location[i] - template_location[self.config.PARENT[i]]))
                                  + joint_location[self.config.PARENT[i]])
        return np.array(joint_location)
