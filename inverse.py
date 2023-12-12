import numpy as np
import scipy
from scipy import optimize


def fit_SLSQP(adsorption, kernel, alpha=0, beta=0, a_array=None):
    def kernel_loading(pore_dist):
        return np.multiply(
            kernel,
            pore_dist[:, np.newaxis]
        ).sum(axis=0)

    def kernel_loading2(pore_dist, a_array):
        p_d = np.empty(len(a_array))
        p_d[:-1] = a_array[1:] - a_array[:1]
        p_d[-1] = p_d[-2]
        p_d = np.multiply(p_d, pore_dist)
        n_s = kernel.T.dot(p_d)
        return n_s

    def sum_squares(pore_dist):
        S_tot = np.sum(pore_dist)
        w = pore_dist / S_tot
        return np.square(
            np.subtract(
                kernel_loading2(pore_dist, a_array),
                adsorption)).sum(axis=0) + alpha * np.sum(w * np.log(w)) / len(pore_dist) + beta * np.sum(
            np.square(pore_dist))

    cons = [{
        'type': 'ineq',
        'fun': lambda x: x,
    }]
    guess = np.array([0.001 for _ in range(len(kernel))])
    bounds = [(0, None) for _ in range(len(kernel))]
    result = optimize.minimize(
        sum_squares,
        guess,
        method='SLSQP',
        bounds=bounds,
        constraints=cons,
        options={'ftol': 1e-04}
    )
    return result


def fit_matrix(adsorption, kernel, rcond=0):
    return np.linalg.lstsq(a=kernel, b=adsorption, rcond=rcond)

