import inspect
import numpy as np

from methods import (
    gradientDescentConst,
    gradientDescentConstQuickGolden,
    gradientMethodQuickFibonacci,
    conjugateGradientDescent,
    conjugateDirectionMethod,
    newtonMethod,
    limited,
)

from plot import showWithTrajectory


def checkResult(arr, result, eps):
    for i in range(len(arr)):
        if abs(arr[i] - result[i]) > eps:
            return False
    return True


def show(trajectory, result, eps, title):
    if not trajectory:
        print(f"-- {title} diverges", title)
    n = len(trajectory) - 1
    if n >= limited:
        return
    print(f"+ {title} {n}" if checkResult(trajectory[len(trajectory) - 1], result, eps) else f"-- {title} {n}")


def test(fn, arg, eps, step, alpha, result):
    print(inspect.getsource(fn))
    show(gradientDescentConst(fn, arg, eps, step), result, eps, "DESCENT const step")
    show(gradientDescentConst(fn, arg, eps, step, alpha), result, eps, "DESCENT ratio")
    show(gradientDescentConstQuickGolden(fn, arg, eps), result, eps, "QUICK DESCENT golden")
    show(gradientMethodQuickFibonacci(fn, arg, eps), result, eps, "QUICK DESCENT fibonacci")
    show(conjugateGradientDescent(fn, arg, eps), result, eps, "CONJUGATED GRADS")
    show(conjugateDirectionMethod(fn, arg, eps), result, eps, "CONJUGATED DIRS")
    show(newtonMethod(fn, arg, eps), result, eps, "NEWTON")
    print()


def draw(fn, trajectory, title):
    showWithTrajectory(
        fn, trajectory,
        min(point[0] for point in trajectory) - 1,
        max(point[0] for point in trajectory) + 1,
        min(point[1] for point in trajectory) - 1,
        max(point[1] for point in trajectory) + 1,
        title=title
    )


def drawTest(fn, arg, eps, step, alpha):
    draw(fn, gradientDescentConst(fn, arg, eps, step), "DESCENT const step")
    draw(fn, gradientDescentConst(fn, arg, eps, step, alpha), "DESCENT const ratio")
    draw(fn, gradientDescentConstQuickGolden(fn, arg, eps), "QUICK DESCENT golden")
    draw(fn, gradientMethodQuickFibonacci(fn, arg, eps), "QUICK DESCENT fibonacci")
    draw(fn, conjugateGradientDescent(fn, arg, eps), "CONJUGATE GRAD")
    draw(fn, conjugateDirectionMethod(fn, arg, eps), "CONJUGATE DIRS")
    draw(fn, newtonMethod(fn, arg, eps), "NEWTON")


def testAndPlot(fn, x, eps, step, alpha, answer):
    test(fn, x, eps, step, alpha, answer)
    drawTest(fn, x, eps, step, alpha)


def main():
    fnFirst = lambda x, y: x ** 2 + y ** 2
    fnSecond = lambda x, y: 22 * ((x - 100) ** 4) + 8 * (y ** 4)
    fnThird = lambda x, y: x ** 4 + y ** 4 + 2 * x * x * y * y - 4 * x + 3
    test(fnFirst, np.array([70., 50.]), 0.001, 5., 0.95, np.array([0., 0.]))
    testAndPlot(fnThird, np.array([100., 100.]), 0.001, 3., 0.95, np.array([1., 0.]))


if __name__ == '__main__':
    main()
