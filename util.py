import os
import numpy as np


def read_data(error=0, is_train=True):
    """
    Args:
        error (int): The index of error, 0 means normal data
        is_train (bool): Read train or test data
    Returns:
        data and labels
    """
    fi = os.path.join('data/', 
        ('d0' if error < 10 else 'd') + str(error) + ('.dat' if is_train else '_te.dat'))
    data = np.fromfile(fi, dtype=np.float32, sep='   ')
    if fi == 'data/d00.dat':
        data = data.reshape(-1, 500).T
    else:
        data = data.reshape(-1, 52)
    # if not is_train:
    #     data = data[160: ]
    return data, np.ones(data.shape[0], np.float32) * error
