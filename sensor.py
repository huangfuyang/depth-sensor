import numpy as np


def cal_angle_three_point(p1,p2,p3):
    v1,v2 = p2-p1,p3-p2
    if np.linalg.norm(v1) != 0 and np.linalg.norm(v2) != 0:
        v1 = v1 / np.linalg.norm(v1)
        v2 = v2 / np.linalg.norm(v2)
        return np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))
    return 0


# joint  sequence:
# wrist, index_mcp, index_pip, index_dip, index_tip, middle_mcp, middle_pip, middle_dip, middle_tip, ring_mcp,
# ring_pip, ring_dip, ring_tip, little_mcp, little_pip, little_dip, little_tip, thumb_mcp, thumb_pip,
# thumb_dip, thumb_tip
def cal_angle_from_pos(labels):
    p_labels = labels.reshape(-1,3)
    # project to y-z plane
    p_labels[:,0] = 0
    angles = np.zeros(15,dtype=np.float32)

    assert p_labels.shape[0] == 21
    for i in range(5):
        angles[i * 3] = cal_angle_three_point(p_labels[0],p_labels[i*4+1],p_labels[i*4+2])
        angles[i * 3+1] = cal_angle_three_point(p_labels[i*4+1],p_labels[i*4+2],p_labels[i*4+3])
        angles[i * 3+2] = cal_angle_three_point(p_labels[i*4+2],p_labels[i*4+3],p_labels[i*4+4])
    return angles/np.pi
