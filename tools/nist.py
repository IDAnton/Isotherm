import numpy as np


class Reader:
    def __init__(self):
        pass

    def clean_txt(self, exp_file):
        exp_data = np.loadtxt(f"../data/real/{exp_file}.txt", delimiter=",")
        pressure = exp_data.T[1]
        n_s = exp_data.T[3]
        result = np.stack((pressure, n_s), axis=1)
        np.savetxt(f"../data/real/{exp_file}_processed.txt", result)


if __name__ == "__main__":
    r = Reader()
    r.clean_txt("MIL-101_2")
