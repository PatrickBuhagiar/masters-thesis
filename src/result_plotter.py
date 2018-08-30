import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import csv
import numpy as np


X = np.arange(5, 11, 1)  # number of nodes
Y = np.arange(0.0005, 0.0021, 0.0001)  # learning rates

accuracies = np.array(list(csv.reader(open("base_out/correct/base_accuracies_5-11_0005-0021.csv"), delimiter=","))).astype(
    "float")
f1s = np.array(list(csv.reader(open("base_out/correct/base_f1s_5-11_0005-0021.csv"), delimiter=","))).astype("float")
Y, X = np.meshgrid(Y, X)

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_xlabel("Learning Rate")
ax.set_ylabel("Number of Hidden Layer Nodes")
ax.set_zlabel("Accuracy")

max_z=max(accuracies)
index_max_z=accuracies.argmax()

plt.title("Base Case - 3D plot for Number of Nodes VS Learning Rate VS Accuracy")
surf = ax.plot_surface(Y, X, accuracies, cmap=cm.coolwarm, rstride=1, cstride=1, linewidth=0, label="accuracy", antialiased=False)
# surf = ax.plot_surface(Y, X, f1s, cmap=cm.coolwarm, rstride=1, cstride=1, linewidth=0, label="f1")
plt.show()
