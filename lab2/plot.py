import matplotlib.pyplot as plt
import numpy as np


def showWithTrajectory(f, trajectory, leftX, rightX, leftY, rightY, scalar=10, title="F(X, Y)"):
    a = plt.figure().add_subplot(111, projection="3d")
    a.set_xlabel("X")
    a.set_ylabel("Y")
    a.set_title(title)
    x = np.linspace(leftX, rightX, int((rightX - leftX) * scalar))
    y = np.linspace(leftY, rightY, int((rightY - leftY) * scalar))
    x, y = np.meshgrid(x, y)
    z = f(x, y)
    a.contour3D(x, y, z)
    a.scatter(
        [point[0] for point in trajectory],
        [point[1] for point in trajectory],
        [f(*point) for point in trajectory],
        c="red"
    )
    plt.show()
