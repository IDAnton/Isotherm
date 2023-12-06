import numpy as np
import scipy
from scipy import optimize


def fit_SLSQP(adsorption, kernel, alpha=0, beta=0):
    def kernel_loading(pore_dist):
        return np.multiply(
            kernel,
            pore_dist[:, np.newaxis]
        ).sum(axis=0)

    def sum_squares(pore_dist):
        S_tot = np.sum(pore_dist)
        w = pore_dist / S_tot
        return np.square(
            np.subtract(
                kernel_loading(pore_dist),
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

