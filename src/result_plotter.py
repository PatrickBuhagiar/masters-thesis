import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import csv
import numpy as np


X = np.arange(30, 36, 1)  # number of nodes
Y = np.arange(0.006, 0.02, 0.002)  # learning rates

accuracies = np.array(list(csv.reader(open("h2_accuracies_26-36_002-007.csv"), delimiter=","))).astype(
    "float")
f1s = np.array(list(csv.reader(open("h2_f1s_26-36_002-007.csv"), delimiter=","))).astype("float")
Y, X = np.meshgrid(Y, X)

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_xlabel("Learning Rate")
ax.set_ylabel("Number of Hidden Layer Nodes")
ax.set_zlabel("Accuracy")

plt.title("H2 - 3D plot for Number of Nodes VS Learning Rate VS Accuracy")
surf = ax.plot_surface(Y, X, accuracies, cmap=cm.coolwarm, rstride=1, cstride=1, linewidth=0, label="accuracy", antialiased=False)
surf = ax.plot_surface(Y, X, f1s, cmap=cm.coolwarm, rstride=1, cstride=1, linewidth=0, label="f1")
plt.show()
