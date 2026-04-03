import logging
import os
from argparse import ArgumentParser, RawTextHelpFormatter
from time import time

import numpy as np
from scipy.sparse.linalg import expm_multiply

from operators import load_operator
from states import coherent_state
from utils import parse


def create_logger(filename):
    logger = logging.getLogger()
    file_handler = logging.FileHandler(filename)
    console_handler = logging.StreamHandler()

    logger.setLevel(logging.INFO)
    file_handler.setLevel(logging.INFO)
    console_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def evolution(psi, H, parameters, model,
              operators=None, logger=None, split=False):
    '''
    An auxiliary function called by `run` to perform the computation. See the
    docstring of `run` for a description of the arguments.
    '''
    j_0, c_0, t, j_max, T, steps, _ = parameters

    if operators is None:
        operators = ['p_x', 'p_z', 'c_x', 'c_z', 's_x', 's_z',
                     'V', 'V_inv', 'N_jmin', 'N_jmax', 'N_jmax-1']

    operators = [(name, load_operator(name, j_max, logger=logger))
                 for name in operators]

    parameters_str = '__'.join(str(x)[:8] for x in parameters[:-1])
    directory = f'data/{j_max}/{model}/{parameters_str}'
    os.makedirs(directory, exist_ok=True)

    def write_ev(state, operator, name, t):
        Opsi = operator @ state
        O_t = np.real(np.dot(state.conj(), Opsi))
        O2_t = np.linalg.norm(Opsi)**2
        with open(f'{directory}/{name}.txt', 'a') as f:
            f.write(f'{t} {O_t} {O2_t}\n')

    def write_norm(psi, t):
        with open(f'{directory}/norm.txt', 'a') as f:
            f.write(f'{t} {np.linalg.norm(psi)}\n')

    def split_(H, state):
        idx = np.arange(j_max**3)
        ix, iy, iz = np.unravel_index(idx, (j_max, j_max, j_max), order='C')
        idx_odd = idx[(ix + iy + iz) % 2 == 0]
        idx_even = idx[(ix + iy + iz) % 2 == 1]
        H_odd = H[idx_odd][:, idx_odd].tocsr()
        H_even = H[idx_even][:, idx_even].tocsr()
        psi_odd = psi[idx_odd]
        psi_even = psi[idx_even]
        return H_odd, H_even, psi_odd, psi_even, idx_odd, idx_even

    def combine(psi_odd, psi_even, idx_odd, idx_even):
        psi = np.zeros(j_max**3, dtype=np.complex128)
        psi[idx_odd] = psi_odd
        psi[idx_even] = psi_even
        return psi

    if logger is not None:
        logger.info('t = 0')

    write_norm(psi, 0)
    write_ev(psi, H, 'H', 0)

    for name, matrix in operators:
        with open(f'{directory}/{name}.txt', 'w') as f:
            f.write(f'# Operator: {name}\n')
            f.write(f'# {j_0 = }, {c_0 = }, {t = }, {j_max = }, '
                    f'{T = }, {steps = }\n')
        write_ev(psi, matrix, name, 0)

    if split:
        H_odd, H_even, psi_odd, psi_even, idx_odd, idx_even = split_(H, psi)

    dt = T / steps
    for i in range(steps):
        t = round((i + 1) * dt, 6)
        if logger is not None:
            logger.info(f'{t = }')
        if split:
            psi_odd = expm_multiply(-1j * H_odd * dt, psi_odd)
            psi_even = expm_multiply(-1j * H_even * dt, psi_even)
            psi = combine(psi_odd, psi_even, idx_odd, idx_even)
        else:
            psi = expm_multiply(-1j * H * dt, psi)
        write_norm(psi, t)
        write_ev(psi, H, 'H', t)
        for name, matrix in operators:
            write_ev(psi, matrix, name, t)


def run(parameters, H, model, operators=None, split=False):
    '''
    Runs the computation defined by the given parameters and Hamiltonian H. The
    parameters should be given in the form (j_0, c_0, t, j_max, T, steps, _),
    where the individual parameters are:

    - j_0, c_0, t: parameters defining the initial state
    - j_max: cutoff
    - T: length of the time interval
    - steps: number of time steps
    - _: a parameter used only by the Julia program (its value is ignored here)
      but is passed to both programs for consistency

    `model` is a string which describes the Hamiltonian and is generated
    automatically by the main function before calling `run`.

    Optionally, a list of operator names can be passed, whose expectation
    values and dispersions will be computed at each time step. By default the
    following operators are considered: p_x, p_z, c_x, c_z, s_x, s_z, V, V_inv,
    N_jmax, N_jmax-1, N_jmin, and the Hamiltonian.

    If `split=True` is passed, the state vector and the Hamiltonian are split
    into their projections onto the 'even' and 'odd' subspaces as described at
    the end of section 4.1, and the action of the time evolution operator on
    the state is computed separately in each subspace.
    '''
    j_0, c_0, t, j_max, T, steps, _ = parameters
    global logger

    parameters_str = f'parameters {j_0 = }, {c_0 = }, t = {t:.6g}, ' \
                     f'{j_max = }, {T = }, {steps = }'
    logger.info(f'Starting computation for {parameters_str}. Model: {model}')

    psi_0 = coherent_state(j_0, c_0, t, j_max)

    t0 = time()

    try:
        evolution(psi_0, H, parameters, model,
                  operators=operators, logger=logger, split=split)
    except Exception as e:
        logger.error(e, exc_info=True)
        outcome = 'Computation terminated due to error for {parameters_str}'
    else:
        outcome = f'Computation completed for {parameters_str}'
    finally:
        logger.info(f'{outcome}. Model: {model}. '
                    f'Elapsed time {time() - t0:.3f} seconds.')


def main():
    parser = ArgumentParser(formatter_class=RawTextHelpFormatter)
    parser.add_argument(
        'parameters',
        nargs=7,
        metavar=('j_0', 'c_0', 't', 'j_max', 'T', 'steps', 'm'),
        help='- j_0, c_0, t: parameters defining the initial state\n'
        '- j_max: cutoff\n'
        '- T: length of the time interval\n'
        '- steps: number of time steps\n'
        '- m: a parameter used only by the Julia program; its value is\n'
        '  ignored by the Python program but must be passed for consistency'
    )
    parser.add_argument(
        '--beta',
        type=float,
        default=0,
        metavar='beta',
        help='Consider the Lorentzian model with a given value of the\n'
        'Barbero-Immirzi parameter. By default the Euclidean model is\n'
        'considered'
    )
    parser.add_argument(
        '--abc',
        nargs=4,
        metavar=('kind', 'alpha', 'beta', 'gamma'),
        help="Consider one of the 'toy' Hamiltonians defined by Eq. (6.1)\n"
        '(kind=1) or (6.2) (kind=2) for given parameters alpha, beta, gamma'
    )

    args = parser.parse_args()
    parameters = [parse(arg) for arg in args.parameters]
    j_max = parameters[3]

    if not args.abc:
        model = 'H_E'
        H = load_operator('H_E', j_max, logger=logger)
        if args.beta:
            beta = args.beta
            model = f'H_L/{beta=}'
            H_L = load_operator('H_L', j_max, logger=logger)
            H = 1 / beta**2 * H + (1 + beta**2) / beta**2 * H_L
            del H_L
    else:
        kind, a, b, c = [parse(arg) for arg in args.abc]
        model = f'H_abc/{kind}__{a:.4g}__{b:.4g}__{c:.4g}'
        H = load_operator('H_abc', j_max, logger=logger,
                          kind=kind, a=a, b=b, c=c)

    run(parameters, H, model, split=not args.beta)


if __name__ == '__main__':
    logger = create_logger('output.log')
    main()
