import numpy as np
import vtk
import json
import math
from vtkmodules.util import numpy_support

class SMART_GARMENT:

    def __init__(self, config_file_path: str, obj_file_path: str):
        '''
        
        :param config_file_path: IO & ADC layout of smart garment
        :param obj_file_path: .obj 3D human model object
        '''
        with open(config_file_path) as config_file:
            self.config = json.load(config_file)["normal"]
        
        self.obj = OBJ_MODEL(obj_file_path)

    def set_pressure_data(self, cloths_data: np.ndarray, pants_data: np.ndarray):
        '''

        :param cloths_data: cloths pressure data, ndarray in shape [56, 40]
        :param pants_data: pants pressure data, ndarray in shape [64, 32]
        '''
        self.cloths_data = cloths_data
        self.pants_data = pants_data

    def make_vtk_scalar(self):
        '''
        
        make scalar for vtk using self.cloths_data and self.pants_data
        :return: vtkFloatArray used to color .obj model in VTK
        '''
        phi_z_list = self.obj.get_phi_z_all_vertices()
        scalars = vtk.vtkFloatArray()

        for item in phi_z_list:

            if(item[0] == 'none'):
                scalars.InsertNextValue(0)
            
            elif(item[0] == 'left_leg' or item[0] == 'right_leg'):
                '''
                
                pants config have prefix 'left_leg' or 'right_leg'
                phi & z decide which channel to select
                '''
                IO = self.config[item[0]]["horizontal"][math.floor(item[2] * len(self.config[item[0]]["horizontal"]))]
                ADC = self.config[item[0]]["vertical"][math.floor(item[1] * len(self.config[item[0]]["vertical"]))]
                scalars.InsertNextValue(self.pants_data[IO][ADC] + 50)
            else:
                IO = self.config[item[0]]["horizontal"][math.floor(item[2] * len(self.config[item[0]]["horizontal"]))]
                ADC = self.config[item[0]]["vertical"][math.floor(item[1] * len(self.config[item[0]]["vertical"]))]
                if(IO >= 48 and IO < 52 and ADC >= 8 and ADC < 16):
                    '''
                    
                    front half of body have 4 horizontal stripes less than back half
                    '''
                    scalars.InsertNextValue(0)
                else:
                    scalars.InsertNextValue(self.cloths_data[IO][ADC] + 50)
        return scalars

class OBJ_MODEL:

    def __init__(self, obj_file_path: str):
        '''
        
        :param obj_file_path: path of .obj model
        :var   reader: vtkOBJReader to load .obj file
        :var   vertices: ndarray containing location of vertices on OBJ model
        '''
        self.reader = vtk.vtkOBJReader()
        self.reader.SetFileName(obj_file_path)
        self.reader.Update()
        self.vertices = numpy_support.vtk_to_numpy(self.reader.GetOutput().GetPoints().GetData())
        '''
        
        five local cylinders
        '''
        self.left_arm = ARM('left')
        self.right_arm = ARM('right')
        self.left_leg = LEG('left')
        self.right_leg = LEG('right')
        self.body = BODY()

    def get_area(self, vertex: np.ndarray):
        '''
        
        locate local cylinder of 1 vertex
        :param vertex: 3D coordinate of 1 vertex
        :return: name of local cylinder input vertex belongs to
        '''
        if(vertex[1] > 156):
            return 'head'

        elif((abs(vertex[0]) > 20 and vertex[1] < 110) or (abs(vertex[0]) > 16 and vertex[1] >= 110)):
            if(vertex[0] > 0):
                return 'left_arm'
            else:
                return 'right_arm'
            
        elif(vertex[1] > 110):
            if(vertex[1] > 150 and vertex[2] > -5):
                return 'none'
            else:
                return 'body'
            
        else:
            if(vertex[0] > 0):
                return 'left_leg'
            else:
                return 'right_leg'
            
    def check_phi_z_valid(self, phi: float, z: float):
        '''
        
        check if phi and z result is legel
        :param phi: phi coordinate in local cylinder
        :param z: z coordinate in local cylinder
        '''
        return (z >= 0 and z <= 1 and phi >=0 and phi <= 1)
    
    def get_phi_z_all_vertices(self):
        '''
        
        calculate phi, z value of all vertices
        phi and z are shaped into 0-1 ratio form
        :return: ndarray containing (cylinder id, phi, z) value of all vertices on .obj model
        '''
        phi_z_list = []

        for point in self.vertices:

            area = self.get_area(point)

            if(area == 'body'):
                phi, z = self.body.get_phi_z(point)
                if(self.check_phi_z_valid(phi, z)):
                    phi_z_list.append(['body', phi, z])
                else:
                    phi_z_list.append(['none', -1, -1])

            elif(area == 'left_arm'):
                phi, z = self.left_arm.get_phi_z(point)
                if(self.check_phi_z_valid(phi, z)):
                    phi_z_list.append(['left_arm', phi, z])
                else:
                    phi_z_list.append(['none', -1, -1])

            elif(area == 'right_arm'):
                phi, z = self.right_arm.get_phi_z(point)
                if(self.check_phi_z_valid(phi, z)):
                    phi_z_list.append(['right_arm', phi, z])
                else:
                    phi_z_list.append(['none', -1, -1])

            elif(area == 'left_leg'):
                phi, z = self.left_leg.get_phi_z(point)
                if(self.check_phi_z_valid(phi, z)):
                    phi_z_list.append(['left_leg', phi, z])
                else:
                    phi_z_list.append(['none', -1, -1])

            elif(area == 'right_leg'):
                phi, z = self.right_leg.get_phi_z(point)
                if(self.check_phi_z_valid(phi, z)):
                    phi_z_list.append(['right_leg', phi, z])
                else:
                    phi_z_list.append(['none', -1, -1])

            else:
                phi_z_list.append(['none', -1, -1])

        return phi_z_list

class ARM:

    def __init__(self, mode: str):
        '''
        
        arm local cylinder
        (start, end) represent the axis of ARM
        values of start & end are set according to OBJ human model we use
        reference vector is set according to the slope of axis (start, end)
        :param mode: str indicates 'left' or 'right' arm
        :var   start: the lower segmentation point on OBJ model
        :var   end: the higher segmentation point on OBJ model
        :var   reference: reference vector for phi calculation
        '''
        if(mode == 'left'):
            self.start = np.array([30, 100, 0])
            self.end = np.array([19, 148, 0])
            slope = (self.end[1] - self.start[1]) / (self.end[0] - self.start[0])
            self.reference = np.array([-1, 1/slope, 0])
            self.reference = self.reference / np.linalg.norm(self.reference)

        elif(mode == 'right'):
            self.start = np.array([-30, 100, 0])
            self.end = np.array([-19, 148, 0])
            slope = (self.end[1] - self.start[1]) / (self.end[0] - self.start[0])
            self.reference = np.array([1, -1/slope, 0])
            self.reference = self.reference / np.linalg.norm(self.reference)

        else:
            raise ValueError("param mode must be 'left' or 'right'.")
        
    def get_phi_z(self, input_point: np.ndarray):
        '''
        
        calculate phi, z value of input_point on local cylinder
        :param input_point: ndarray of coordinate (x, y, z) in global system
        :return: tupe contains (phi, z) value
        '''
        if(type(input_point) != np.ndarray):
            input_point = np.array(input_point)
        axis = self.end - self.start
        axis = axis / np.linalg.norm(axis)
        '''
        
        z is the projection length of vector [input_point, self.start] on axis
        '''
        z = np.dot(input_point - self.start, axis)
        '''
        
        phi is the angle that reference rotate to vec counter-clockwise
        '''
        vec = (input_point - self.start) - z * axis
        vec = vec / np.linalg.norm(vec)
        cosPhi = np.dot(self.reference, vec)
        sign = np.sign(np.cross(self.reference, vec)[1])
        Phi = np.arccos(cosPhi) if sign >= 0 else (2 * np.pi - np.arccos(cosPhi))
        return (Phi / (2 * np.pi), z / np.linalg.norm(self.end - self.start))

class LEG:

    def __init__(self, mode: str):
        '''
        
        leg local cylinder
        (start, end) represent the axis of LEG
        values of start & end are set according to OBJ human model we use
        reference vector is set according to the slope of axis (start, end)
        :param mode: str indicates 'left' or 'right' leg
        :var   start: the lower segmentation point on OBJ model
        :var   end: the higher segmentation point on OBJ model
        :var   reference: reference vector for phi calculation
        '''
        if(mode == 'left'):
            self.start = np.array([10, 15, 0])
            self.end = np.array([10, 110, 0])
            self.reference = np.array([-1, 0, 0])

        elif(mode == 'right'):
            self.start = np.array([-10, 15, 0])
            self.end = np.array([-10, 110, 0])
            self.reference = np.array([1, 0, 0])

        else:
            raise ValueError("mode must be 'left' or 'right'.")
        
    def get_phi_z(self, input_point: np.ndarray):
        '''
        
        calculate phi, z value of input_point on local cylinder
        :param input_point: ndarray of coordinate (x, y, z) in global system
        :return: tupe contains (phi, z) value
        '''
        if(type(input_point) != np.ndarray):
            input_point = np.array(input_point)
        axis = self.end - self.start
        axis = axis / np.linalg.norm(axis)
        '''
        
        z is the projection length of vector [input_point, self.start] on axis
        '''
        z = np.dot(input_point - self.start, axis)
        '''
        
        phi is the angle that reference rotate to vec counter-clockwise
        '''
        vec = (input_point - self.start) - z * axis
        vec = vec / np.linalg.norm(vec)
        cosPhi = np.dot(self.reference, vec)
        sign = np.sign(np.cross(self.reference, vec)[1])
        Phi = np.arccos(cosPhi) if sign >= 0 else (2 * np.pi - np.arccos(cosPhi))
        return (Phi / (2 * np.pi), z / np.linalg.norm(self.end - self.start))
    
class BODY:

    def __init__(self):
        '''
        
        body local cylinder
        (start, end) represent the axis of BODY
        values of start & end are set according to OBJ human model we use
        reference vector is set according to the slope of axis (start, end)
        :var   start: the lower segmentation point on OBJ model
        :var   end: the higher segmentation point on OBJ model
        :var   reference: reference vector for phi calculation
        '''
        self.start = np.array([0, 110, 0])
        self.end = np.array([0, 156, 0])
        self.reference = np.array([1, 0, 0])

    def get_phi_z(self, input_point: np.ndarray):
        '''
        
        calculate phi, z value of input_point on local cylinder
        :param input_point: ndarray of coordinate (x, y, z) in global system
        :return: tupe contains (phi, z) value
        '''
        if(type(input_point) != np.ndarray):
            input_point = np.array(input_point)
        axis = self.end - self.start
        axis = axis / np.linalg.norm(axis)
        '''
        
        z is the projection length of vector [input_point, self.start] on axis
        '''
        z = np.dot(input_point - self.start, axis)
        '''
        
        phi is the angle that reference rotate to vec counter-clockwise
        '''
        vec = (input_point - self.start) - z * axis
        vec = vec / np.linalg.norm(vec)
        cosPhi = np.dot(self.reference, vec)
        sign = np.sign(np.cross(self.reference, vec)[1])
        Phi = np.arccos(cosPhi) if sign >= 0 else (2 * np.pi - np.arccos(cosPhi))
        return (Phi / (2 * np.pi), z / np.linalg.norm(self.end - self.start))
