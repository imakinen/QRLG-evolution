import os
import sys
from fractions import Fraction
from scipy.io import mmwrite

from operators import load_operator


def convert(args):
    '''
    Converts an operator constructed by `operators.create_operator` to a format
    readable by the Julia program.
    '''
    name, j_max = args[1], int(args[2])

    if name == 'H_abc':
        kind = int(args[3])
        a, b, c = [float(Fraction(x)) for x in args[4:]]
        A = load_operator(name, j_max, save=False, kind=kind, a=a, b=b, c=c)
    else:
        A = load_operator(name, j_max)

    os.makedirs(f'data/julia/{j_max}/operators', exist_ok=True)
    mmwrite(f'data/julia/{j_max}/operators/{name}.mtx', A)


if __name__ == '__main__':
    convert(sys.argv)
