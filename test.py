import numpy as np
import matplotlib.pyplot as plt
import sympy
import math
import time

a = np.broadcast_to(np.zeros(2)[:, None], (2, 130))
num_dim = 2
time_step = 0.01
t = 1.3
shape = (num_dim, int(t/time_step))  # Shape for each input (num_dim, t/time_step)
print(shape)