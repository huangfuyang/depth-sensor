# dataset params
DATA_EXT = '_depth.bin'
IMG_EXT = '_depth.jpg'
LABEL = 'joint.txt'
GESTURES = ['1']
# GESTURES = ['1','2','3','4','5','6','7','8','9','I','IP','L','MP','RP','T','TIP','Y']
GESTURES_LEN = len(GESTURES)
MAX_SAMPLE_LEN = 500
JOINT_LEN = 21
JOINT_POS_LEN = JOINT_LEN *3

# tsdf params
VOXEL_RES = 32  # M
TRUC_DIS_T = 3

# camera params
FOCAL = 241.42
CENTER_X = 160
CENTER_Y = 120

#train params
LEARNING_RATE = 0.01
MOMENTUM = 0.9
EPOCH_COUNT = 5
WEIGHT_DECAY = 0.0005