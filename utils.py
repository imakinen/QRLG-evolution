def index(spins, j_max):
    '''
    Returns the one-dimensional index corresponding to the basis state with
    spins = (j_x, j_y, j_z) for a given cutoff j_max.
    '''
    j_x, j_y, j_z = spins
    return j_max**2 * (j_x - 1) + j_max * (j_y - 1) + (j_z - 1)


def spins(index, j_max):
    '''
    Returns the spins labeling the basis state |j_x, j_y, j_z> corresponding to
    a given index and cutoff j_max.
    '''
    j_x = index // j_max**2 + 1
    r = index % j_max**2
    j_y = r // j_max + 1
    j_z = r % j_max + 1
    return j_x, j_y, j_z


def parse(arg):
    '''
    Parses a numerical argument passed on the command line as an integer, a
    decimal number or a fraction.
    '''
    if '/' in arg:
        num, den = arg.split('/')
        return parse(num) / parse(den)
    return float(arg) if '.' in arg else int(arg)
