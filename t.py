import matplotlib.pylab as plt

x = []
y = []

x.append(1)
y.append(4)
x.append(2)
y.append(5)

x, = plt.plot(x, label="X")
y, = plt.plot(y, label="Y")
plt.legend(handles = [x, y])
plt.show()