import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import csv
import numpy as np

X = np.arange(30, 41, 2)  # number of nodes
Y = np.arange(0.0006, 0.0051, 0.0004)  # learning rates

accuracies = np.array(list(csv.reader(open("h3_accuracies_30-41_0006-0051.csv"), delimiter=","))).astype(
    "float")
f1s = np.array(list(csv.reader(open("h3_f1s_30-41_0006-0051.csv"), delimiter=","))).astype("float")
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
