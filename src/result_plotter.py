import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import csv
import numpy as np

X = np.arange(36, 56, 2)  # number of nodes
Y = np.arange(0.00001, 0.00006, 0.00001)  # learning rates

accuracies = np.array(list(csv.reader(open("h2_accuracies_36-56_00001-00006.csv"), delimiter=","))).astype(
    "float")
f1s = np.array(list(csv.reader(open("h2_f1s_36-56_00001-00006.csv"), delimiter=","))).astype("float")
Y, X = np.meshgrid(Y, X)

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_xlabel("Learning Rate")
ax.set_ylabel("Number of Hidden Layer Nodes")
ax.set_zlabel("F1 Score")

plt.title("Base Case - 3D plot for Number of Nodes VS Learning Rate VS F1 Score")
surf = ax.plot_surface(Y, X, accuracies, cmap=cm.coolwarm, rstride=1, cstride=1, linewidth=0, label="accuracy")
# surf = ax.plot_surface(Y, X, f1s, cmap=cm.coolwarm, rstride=1, cstride=1, linewidth=0, label="f1")
plt.show()
