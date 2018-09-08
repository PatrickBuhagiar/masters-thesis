import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import csv
import numpy as np

X = np.arange(40, 61, 1)  # number of nodes
Y = np.arange(0.000001, 0.00001, 0.000001)

accuracies = np.array(list(csv.reader(open("h3_accuracies_40-61_000005-00005.csv"), delimiter=","))).astype(
    "float")
f1s = np.array(list(csv.reader(open("h3_f1s_40-61_000005-00005.csv"), delimiter=","))).astype("float")
Y, X = np.meshgrid(Y, X)

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_xlabel("Learning Rate")
ax.set_ylabel("Number of Hidden Layer Nodes")
ax.set_zlabel("F1 Score")

plt.title("Hypothesis 3 - 3D plot for Number of Nodes VS Learning Rate VS F1 Score")
# surf = ax.plot_surface(Y, X, accuracies, cmap=cm.coolwarm, rstride=1, cstride=1, linewidth=0, label="accuracy")
surf = ax.plot_surface(Y, X, f1s, cmap=cm.coolwarm, rstride=1, cstride=1, linewidth=0, label="f1")
plt.show()
