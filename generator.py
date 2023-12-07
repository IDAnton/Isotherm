import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.animation as animation

# mpl.use('TkAgg')
plt.rcParams.update({'font.size': 14})


# plt.rcParams.update({'font.family': 'Times New Roman'})

class Generator:
    def __init__(self, path_s, path_d, path_p_s, path_p_d, path_a):
        with open(path_s, 'rb') as f:
            self.data_sorb = np.load(f)
        with open(path_d, 'rb') as f:
            self.data_desorb = np.load(f, allow_pickle=True)
        with open(path_p_d, 'rb') as f:
            self.pressures_d = np.load(f)
            self.pressures_d_current = self.pressures_d
        with open(path_p_s, 'rb') as f:
            self.pressures_s = np.load(f)
        with open(path_a, 'rb') as f:
            self.a_array = np.load(f)

        self.pore_distribution = None
        self.n_s = np.zeros(len(self.pressures_s))  # adsorption isotherm data
        self.n_d = np.zeros(len(self.pressures_d))  # desorption isotherm data

    def generate_pore_distribution(self, sigma1, sigma2, d0_1, d0_2, a=1):
        pore_distribution1 = (1 / sigma1) * np.exp(-np.power((self.a_array - d0_1), 2) / (2 * sigma1 ** 2))
        pore_distribution1 /= max(pore_distribution1)
        pore_distribution2 = (1 / sigma2) * np.exp(-np.power((self.a_array - d0_2), 2) / (2 * sigma2 ** 2))
        pore_distribution2 /= max(pore_distribution2)
        self.pore_distribution = pore_distribution1 * a + pore_distribution2
        self.pore_distribution /= max(self.pore_distribution)

    def calculate_isotherms(self):
        self.n_s = np.zeros(len(self.pressures_s))
        self.n_d = np.zeros(len(self.pressures_d))
        for p_i in range(len(self.pressures_s)):
            for d_i in range(len(self.pore_distribution)):
                self.n_s[p_i] += self.pore_distribution[d_i] * self.data_sorb[d_i][p_i]

        for p_i in range(len(self.pressures_d)):
            for d_i in range(len(self.pore_distribution)):
                if not np.isnan(self.data_desorb[d_i][p_i]):
                    self.n_d[p_i] += self.pore_distribution[d_i] * self.data_desorb[d_i][p_i]

    def normalize_data(self):
        self.n_s = self.n_s / self.n_s.max()
        self.n_d = self.n_d / self.n_d.max()

    def interp_desorption(self):
        self.n_d = np.interp(self.pressures_s, self.pressures_d, self.n_d)
        self.pressures_d_current = self.pressures_s

    def plot_isotherm(self):
        fig, axs = plt.subplots(2, 1)
        axs[0].plot(self.pressures_s, self.n_s, marker=".", label="Сорбция")
        axs[0].plot(self.pressures_d_current, self.n_d, marker=".", label="Десорбция")
        axs[0].set_xlabel("Давление")
        axs[0].set_ylabel("Величина адсорбции")
        axs[1].plot(self.a_array, self.pore_distribution, marker=".", label="Размер пор")
        axs[1].set_ylabel("Функция распределения")
        axs[1].set_xlabel("Размер пор (нм)")
        axs[0].legend()
        axs[1].legend()
        plt.show()
        return fig, axs

    def ani(self):
        fig, axs = plt.subplots(2, 1, figsize=(8, 6))
        axs[0].set_ylim(0, 2)
        self.generate_pore_distribution(sigma1=0.1, sigma2=2, d0_1=1, d0_2=10, a=0.1)
        self.calculate_isotherms_from_new_kernel()
        sorb_line, = axs[0].plot(self.pressures_s, self.n_s, marker=".", label="Сорбция")
        desorb_line, = axs[0].plot(self.pressures_d, self.n_d, marker=".", label="Десорбция")
        distr_line, = axs[1].plot(self.a_array, self.pore_distribution, marker=".")

        def animate(i):
            self.generate_pore_distribution(sigma1=0.1, sigma2=2, d0_1=1, d0_2=10, a=i)
            self.calculate_isotherms_from_new_kernel()
            self.normalize_data()
            sorb_line.set_ydata(self.n_s)  # update the data
            desorb_line.set_ydata(self.n_d)  # update the data
            distr_line.set_ydata(self.pore_distribution)  # update the data
            return sorb_line, desorb_line, distr_line

        anim = animation.FuncAnimation(fig, animate, np.linspace(0.1, 10, 250),
                                       interval=25, blit=False)
        writervideo = animation.FFMpegWriter(fps=30)
        anim.save("anim.mp4", writer=writervideo)
        plt.show()

    def generate_data_set(self, name, data_len=3):
        d0_1_range = np.linspace(0.5, 2, data_len)
        d0_2_range = np.linspace(2, 30, data_len)
        sigma1_range = np.linspace(0.1, 1, data_len)
        sigma2_range = np.linspace(1, 10, data_len)
        a_range = np.linspace(0, 2, data_len)
        i = 0
        dataset = []
        for a in a_range:
            for d0_1 in d0_1_range:
                for d0_2 in d0_2_range:
                    for sigma1 in sigma1_range:
                        for sigma2 in sigma2_range:
                            if i % 100 == 0:
                                print(f"generated {i} out of {data_len**5}")
                            i += 1
                            self.generate_pore_distribution(sigma1=sigma1, sigma2=sigma2, d0_1=d0_1, d0_2=d0_2, a=a)
                            self.calculate_isotherms()
                            self.interp_desorption()
                            dataset.append({"isotherm": self.n_s, "a": a, "d0_1": d0_1, "d0_2": d0_2,
                                            "sigma1": sigma1, "sigma2": sigma2, "pore_distribution":self.pore_distribution})
        with open(f'data/datasets/{name}.npy', 'wb') as f:
            np.save(f, np.array(dataset))

    def save_isotherm_and_distribution(self, path):
        np.savez(path, n_s=self.n_s, n_d=self.n_d, distr=self.pore_distribution)


if __name__ == "__main__":
    gen = Generator(path_s="data/initial kernels/Kernel_Carbon_Adsorption.npy",
                              path_d="data/initial kernels/Kernel_Carbon_Desorption.npy",
                              path_p_d="data/initial kernels/Pressure_Carbon.npy",
                              path_p_s="data/initial kernels/Pressure_Carbon.npy",
                              path_a="data/initial kernels/Size_Kernel_Carbon_Adsorption.npy"
                              )
    gen.generate_data_set(data_len=5, name="carbon2")