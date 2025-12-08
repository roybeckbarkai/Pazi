import numpy as np
from distribution_functions import Boltzmann_dis
import matplotlib.pyplot as plt

r = np.linspace(0, 3, 100)

for sigma in [0.1, 1, 1.1]:
    y = Boltzmann_dis(r, sigma, 1)
    plt.plot(r, y, label=f'sigma = {sigma}')

plt.legend()
plt.show()