import os
from functools import partial
from itertools import product

import numpy as np
from scipy.sparse import coo_matrix, load_npz, save_npz

from utils import index


'''
The functions below return the matrix elements of the given operator of the
one-vertex model between the basis states with spins ks = (k_x, k_y, k_z) and
js = (j_x, j_y, j_z).

For operators labeled by a specific coordinate direction, i = 0, 1, 2
corresponds to i = x, y, z respectively.
'''


def p_i(ks, js, i, n=1):
    '''
    The n-th power (default: n=1) of the reduced flux operator p_i:

    (p_i)^n |j_x, j_y, j_z> = (j_i)^n|j_x, j_y, j_z>
    '''
    if ks != js:
        return 0
    return js[i]**n


def c_i(ks, js, i):
    '''
    The symmetric holonomy operator c^{(1)}_i:

    c^{(1)}_x |j_x, j_y, j_z> = 1/2 * (|j_x+1, j_y, j_z> + |j_x-1, j_y, j_z>)
    etc.
    '''
    if any(ks[j] != js[j] for j in range(3) if j != i):
        return 0
    if abs(ks[i] - js[i]) == 1:
        return 1/2
    return 0


def s_i(ks, js, i):
    '''
    The symmetric holonomy operator s^{(1)}_i:

    s^{(1)}_x |j_x, j_y, j_z> = 1/2i * (|j_x+1, j_y, j_z> + |j_x-1, j_y, j_z>)
    etc.
    '''
    if any(ks[j] != js[j] for j in range(3) if j != i):
        return 0
    if ks[i] == js[i] + 1:
        return 1 / 2j
    if ks[i] == js[i] - 1:
        return -1 / 2j
    return 0


def V(ks, js, n=1):
    '''
    The n-th power of the volume operator (default: n=1):

    V^n |j_x, j_y, j_z> = (j_x * j_y * j_z)^(n/2) |j_x, j_y, j_z>
    '''
    if ks != js:
        return 0
    j_x, j_y, j_z = js
    return (j_x * j_y * j_z)**(n/2)


def H_E(ks, js):
    '''
    The Euclidean operator defined by Eq. (3.21)
    '''
    j_x, j_y, j_z = js
    k_x, k_y, k_z = ks

    if k_x == j_x and abs(k_y - j_y) == 1 and abs(k_z - j_z) == 1:
        s = np.sign((k_y - j_y) * (k_z - j_z))
        return s/4 * (j_y * k_y * j_z * k_z / j_x**2)**0.25
    if k_y == j_y and abs(k_x - j_x) == 1 and abs(k_z - j_z) == 1:
        s = np.sign((k_x - j_x) * (k_z - j_z))
        return s/4 * (j_x * k_x * j_z * k_z / j_y**2)**0.25
    if k_z == j_z and abs(k_x - j_x) == 1 and abs(k_y - j_y) == 1:
        s = np.sign((k_x - j_x) * (k_y - j_y))
        return s/4 * (j_x * k_x * j_y * k_y / j_z**2)**0.25

    return 0


def H_L(ks, js):
    '''
    The Lorentzian operator defined by Eq. (3.24)
    '''
    j_x, j_y, j_z = js
    k_x, k_y, k_z = ks

    def f_x(j):
        return j**1.5 / (j_y * j_z)**0.5

    def f_y(j):
        return j**1.5 / (j_x * j_z)**0.5

    def f_z(j):
        return j**1.5 / (j_x * j_y)**0.5

    if any(j == 0 for j in js):
        return 0

    if any(k == 0 for k in ks):
        return 0

    if ks == js:
        return -4 * f_x(j_x) - 4 * f_y(j_y) - 4 * f_z(j_z) \
               - f_x(j_x)**0.5 * (f_x(j_x + 1)**0.5 + f_x(j_x - 1)**0.5) \
               - f_y(j_y)**0.5 * (f_y(j_y + 1)**0.5 + f_y(j_y - 1)**0.5) \
               - f_z(j_z)**0.5 * (f_z(j_z + 1)**0.5 + f_z(j_z - 1)**0.5)

    if k_y == j_y and k_z == j_z:
        if abs(k_x - j_x) == 1:
            return 2 * (f_x(j_x) * f_x(k_x))**0.25 \
                     * (f_x(j_x)**0.5 + f_x(k_x)**0.5)
        if abs(k_x - j_x) == 2:
            return -f_x(j_x)**0.25 * f_x(k_x)**0.25 * f_x((j_x + k_x) / 2)**0.5

    if k_x == j_x and k_z == j_z:
        if abs(k_y - j_y) == 1:
            return 2 * (f_y(j_y) * f_y(k_y))**0.25 \
                     * (f_y(j_y)**0.5 + f_y(k_y)**0.5)
        if abs(k_y - j_y) == 2:
            return -f_y(j_y)**0.25 * f_y(k_y)**0.25 * f_y((j_y + k_y) / 2)**0.5

    if k_x == j_x and k_y == j_y:
        if abs(k_z - j_z) == 1:
            return 2 * (f_z(j_z) * f_z(k_z))**0.25 \
                     * (f_z(j_z)**0.5 + f_z(k_z)**0.5)
        if abs(k_z - j_z) == 2:
            return -f_z(j_z)**0.25 * f_z(k_z)**0.25 * f_z((j_z + k_z) / 2)**0.5

    return 0


def H_abc_I(ks, js, a, b, c):
    '''
    The 'toy' Hamiltonian of the 'first kind' defined by Eq. (6.1) for given
    values of alpha, beta and gamma (a, b, c)
    '''
    j_x, j_y, j_z = js
    k_x, k_y, k_z = ks

    if abs(k_x - j_x) == 1 and abs(k_y - j_y) == 1 and k_z == j_z:
        s = np.sign((k_x - j_x) * (k_y - j_y))
        return s/8 * ((j_x * k_x)**(a/2) * (j_y * k_y)**(b/2)
                      + (j_x * k_x)**(b/2) * (j_y * k_y)**(a/2)) * j_z**c

    if abs(k_x - j_x) == 1 and k_y == j_y and abs(k_z - j_z) == 1:
        s = np.sign((k_x - j_x) * (k_z - j_z))
        return s/8 * ((j_x * k_x)**(a/2) * (j_z * k_z)**(b/2)
                      + (j_x * k_x)**(b/2) * (j_z * k_z)**(a/2)) * j_y**c

    if k_x == j_x and abs(k_y - j_y) == 1 and abs(k_z - j_z) == 1:
        s = np.sign((k_y - j_y) * (k_z - j_z))
        return s/8 * ((j_y * k_y)**(a/2) * (j_z * k_z)**(b/2)
                      + (j_y * k_y)**(b/2) * (j_z * k_z)**(a/2)) * j_x**c

    return 0


def H_abc_II(ks, js, a, b, c):
    '''
    The 'toy' Hamiltonian of the 'second kind' defined by Eq. (6.2) for given
    values of alpha, beta and gamma (a, b, c)
    '''
    j_x, j_y, j_z = js
    k_x, k_y, k_z = ks

    def _pow(x, y):
        if x == 0:
            return 0
        return x**y

    if ks == js:
        return - 1/8 * j_x**(a/2) * ((j_x + 1)**(a/2) + _pow(j_x - 1, a/2)) \
                     * (j_y**b * j_z**c + j_y**c * j_z**b) \
               - 1/8 * j_y**(a/2) * ((j_y + 1)**(a/2) + _pow(j_y - 1, a/2)) \
                     * (j_x**b * j_z**c + j_x**c * j_z**b) \
               - 1/8 * j_z**(a/2) * ((j_z + 1)**(a/2) + _pow(j_z - 1, a/2)) \
                     * (j_x**b * j_y**c + j_x**c * j_y**b)

    if abs(k_x - j_x) == 2 and k_y == j_y and k_z == j_z:
        return 1/8 * (j_x * k_x)**(a/4) * ((j_x + k_x) / 2)**(a/2) \
                   * (j_y**b * j_z**c + j_y**c * j_z**b)

    if k_x == j_x and abs(k_y - j_y) == 2 and k_z == j_z:
        return 1/8 * (j_y * k_y)**(a/4) * ((j_y + k_y) / 2)**(a/2) \
                   * (j_x**b * j_z**c + j_x**c * j_z**b)

    if k_x == j_x and k_y == j_y and abs(k_z - j_z) == 2:
        return 1/8 * (j_z * k_z)**(a/4) * ((j_z + k_z) / 2)**(a/2) \
                   * (j_x**b * j_y**c + j_x**c * j_y**b)

    return 0


def N_j(ks, js, j):
    '''
    The 'occupation number' operator of the basis states in which any spin is
    equal to the given value j:

    N_j |j_x, j_y, j_z> = |j_x, j_y, j_z>  if any j_i = j
                          0                otherwise
    '''
    if ks != js:
        return 0
    if any(j_ == j for j_ in js):
        return 1
    return 0


def create_operator(name, j_max, save=True, **kwargs):
    '''
    Returns a sparse matrix (of type scipy.sparse.csr_matrix) representing the
    given operator for a given cutoff j_max. If `save=True` is passed, a copy
    of the matrix is saved to disk for later use.

    The possible values of `name` are:
        p_x, p_y, p_z, c_x, c_y, c_z, s_x, s_y, s_z, V,
        V_inv (the inverse volume operator), H_E, H_L, H_abc,
        N_jmax, N_jmax-1, N_jmin
    In case of H_abc, the parameters `kind`, `a`, `b` and `c` must be passed as
    keyword arguments.
    '''
    if name == 'N_jmax':
        matrix_element = partial(N_j, j=j_max)
    elif name == 'N_jmax-1':
        matrix_element = partial(N_j, j=j_max-1)
    elif name == 'H_abc':
        a, b, c, kind = kwargs['a'], kwargs['b'], kwargs['c'], kwargs['kind']
        func = {1: H_abc_I, 2: H_abc_II}[kind]
        matrix_element = partial(func, a=a, b=b, c=c)
    else:
        matrix_element = _operators[name]

    # Pre-allocate the arrays defining the sparse matrix in COO format.
    # There can be at most 13 non-zero matrix elements per column.
    nnz = 13 * j_max**3
    dtype = np.complex128 if name[0] == 's' else np.float64

    col = np.empty(nnz, dtype=np.int32)
    row = np.empty(nnz, dtype=np.int32)
    data = np.empty(nnz, dtype=dtype)

    # Assign a value of `dj` according to the action of the operator on the
    # basis states. The action of the operator on the state |j_x, j_y, j_z>
    # produces states |k_x, k_y, k_z> in which each spin k_i lies in the range
    # [j_i - dj, j_i + dj].
    if name[0] in 'p V N':
        dj = 0
    elif name[0] in 'c s' or name == 'H_E' or (name == 'H_abc' and kind == 1):
        dj = 1
    elif name == 'H_L' or (name == 'H_abc' and kind == 2):
        dj = 2

    idx = 0
    for js in product(range(1, j_max+1), repeat=3):
        i = index(js, j_max)
        j_x, j_y, j_z = js
        for ks in product(range(max(j_x - dj, 1), min(j_x + dj, j_max) + 1),
                          range(max(j_y - dj, 1), min(j_y + dj, j_max) + 1),
                          range(max(j_z - dj, 1), min(j_z + dj, j_max) + 1)):
            k = index(ks, j_max)
            m = matrix_element(ks, js)
            if m != 0:
                col[idx] = i
                row[idx] = k
                data[idx] = m
                idx += 1

    row = np.resize(row, idx)
    col = np.resize(col, idx)
    data = np.resize(data, idx)

    matrix = coo_matrix((data, (row, col)), shape=(j_max**3, j_max**3)).tocsr()

    if save:
        directory = f'data/{j_max}/operators'
        os.makedirs(directory, exist_ok=True)
        save_npz(f'{directory}/{name}.npz', matrix)

    return matrix


def load_operator(name, j_max, save=True, logger=None, **kwargs):
    '''
    Loads the given operator from disk if available, otherwise calls
    `create_operator` to construct it.

    Optionally a logger object can be passed for log messages.
    '''
    filename = f'data/{j_max}/operators/{name}.npz'
    if name == 'H_abc':
        save = False
    if not os.path.exists(filename):
        if logger is not None:
            logger.info(f'Creating operator {name}')
        return create_operator(name, j_max, save=save, **kwargs)
    else:
        if logger is not None:
            logger.info(f'Loading operator {name}')
        return load_npz(filename)


_operators = {
    'p_x': partial(p_i, i=0),
    'p_y': partial(p_i, i=1),
    'p_z': partial(p_i, i=2),
    'c_x': partial(c_i, i=0),
    'c_y': partial(c_i, i=1),
    'c_z': partial(c_i, i=2),
    's_x': partial(s_i, i=0),
    's_y': partial(s_i, i=1),
    's_z': partial(s_i, i=2),
    'V': V,
    'V_inv': partial(V, n=-1),
    'p_x_inv': partial(p_i, i=0, n=-1),
    'H_E': H_E,
    'H_L': H_L,
    'N_jmin': partial(N_j, j=1),
}
