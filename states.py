import numpy as np
from functools import cache
from utils import spins


@cache
def cs_amplitude(j, j_0, c_0, t):
    '''
    The amplitude of the one-dimensional coherent state (4.14) with given j_0,
    c_0 and semiclassicality parameter t with respect to the basis state |j>.
    '''
    return np.sqrt(2*j + 1) * np.exp(-t * (j - j_0)**2 / 2 - 1j * c_0 * j)


def coherent_state(j_0, c_0, t, j_max):
    '''
    Returns a numpy array representing the state vector of the normalized
    coherent state (4.15) for given parameters j_0, c_0, t and cutoff j_max.
    '''
    psi = np.zeros(j_max**3, dtype=np.complex128)
    for i in range(j_max**3):
        j_x, j_y, j_z = spins(i, j_max)
        psi[i] = cs_amplitude(j_x, j_0, c_0, t) \
                 * cs_amplitude(j_y, j_0, c_0, t) \
                 * cs_amplitude(j_z, j_0, c_0, t)
    return psi / np.linalg.norm(psi)
