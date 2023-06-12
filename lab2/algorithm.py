from math import sqrt

from mathUtilityPac import FibUtils

minimizers = list()
goldenRatio = (3 - sqrt(5)) / 2
step = 0.05


def minimizer(fn):
    minimizers.append(fn)
    return fn


@minimizer
def goldenRatioMethod(f, leftX, rightX, eps):
    left = leftX
    right = rightX
    segment = abs(right - left)
    x1 = left + goldenRatio * segment
    x2 = right - goldenRatio * segment
    y1 = f(x1)
    y2 = f(x2)
    segments = [(left, right)]
    while segment > eps:
        if y1 >= y2:
            left = x1
            x1 = x2
            x2 = right - goldenRatio * (right - left)
            y1 = y2
            y2 = f(x2)
        else:
            right = x2
            x2 = x1
            x1 = left + goldenRatio * (right - left)
            y2 = y1
            y1 = f(x1)
        segment = right - left
        segments.append((left, right))
    return segments


@minimizer
def fibonacciMethod(f, leftX, rightX, eps):
    left = leftX
    right = rightX
    fib = FibUtils.Fibonacci()
    sz = fib.upperBound((right - left) / eps) - 2
    x1 = left + fib.getFibByNumber(sz) / fib.getFibByNumber(sz + 2) * (right - left)
    x2 = left + fib.getFibByNumber(sz + 1) / fib.getFibByNumber(sz + 2) * (right - left)
    y1 = f(x1)
    y2 = f(x2)
    segments = [(left, right)]
    for i in range(2, sz + 3):
        if y1 > y2:
            left = x1
            x1, y1 = x2, y2
            x2 = left + fib.getFibByNumber(sz - i + 2) / fib.getFibByNumber(sz - i + 3) * (right - left)
            y2 = f(x2)
        else:
            right = x2
            x2, y2 = x1, y1
            x1 = left + fib.getFibByNumber(sz - i + 1) / fib.getFibByNumber(sz - i + 3) * (right - left)
            y1 = f(x1)
        segments.append((left, right))
    return segments
