
class OPENPOSE_15_CONFIG:
    '''
    15 openpose kp.
    The class is only used for indexing
    kp: key-point
    '''
    def __init__(self):
        self.ROOT = 0
        self.MIDHIP = 0
        self.NECK = 1
        self.NOSE = 2
        self.LSHOULDER = 3
        self.RSHOULDER = 4
        self.LELBOW = 5
        self.RELBOW = 6
        self.LWRIST = 7
        self.RWRIST = 8
        self.LHIP = 9
        self.RHIP = 10
        self.LKNEE = 11
        self.RKNEE = 12
        self.LANKLE = 13
        self.RANKLE = 14
        self.PARENT = [None, self.MIDHIP, self.NECK, self.NECK, self.NECK, 
                       self.LSHOULDER, self.RSHOULDER, self.LELBOW, self.RELBOW, 
                       self.MIDHIP, self.MIDHIP, self.LHIP, self.RHIP, self.LKNEE, self.RKNEE]
        self.NUM_JOINT = 15

        self.LIMB_BODY = 0
        self.LIMB_NECK = 1
        self.LIMB_LSHOULDER = 2
        self.LIMB_RSHOULDER = 3
        self.LIMB_LMAINARM = 4
        self.LIMB_RMAINARM = 5
        self.LIMB_LFOREARM = 6
        self.LIMB_RFOREARM = 7
        self.LIMB_LHIP = 8
        self.LIMB_RHIP = 9
        self.LIMB_LTHIGH = 10
        self.LIMB_RTHIGH = 11
        self.LIMB_LCRUS = 12
        self.LIMB_RCRUS = 13
        
        self.NUM_LIMB = 14

        self.LIMB_COLOR = [None, 'red', 'red', 'yellow', 'orange', 'yellow', 'orange', 'yellow', 'orange', 'green', 'blue', 'green', 'blue', 'green', 'blue']
        