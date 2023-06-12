"""
Содержит полезные математические методы
"""

import mpmath

mpmath.dps = 7
mpmath.pretty = True
DiffStep = 0.001


def partialDerivative(f, x, i):
    mask = [0] * len(x)
    mask[i] = 1
    return mpmath.diff(f, x, mask)


def secondPartialDerivative(f, x, FirstI, secondI):
    mask = [0] * len(x)
    mask[FirstI] += 1
    mask[secondI] += 1
    return mpmath.diff(f, x, mask)


def gradient(f, x):
    result = x.copy()
    for i in range(len(x)):
        result[i] = partialDerivative(f, x, i)
    return result


def hessianMatrix(f, x):
    return [
        [float(secondPartialDerivative(f, x, i, j)) for j in range(len(x))]
        for i in range(len(x))
    ]


def norm(x):
    return sum(map(lambda num: num ** 2, x)) ** 0.5


def normalize(x):
    y = x.copy()
    size = norm(y)
    for i in range(len(y)):
        y[i] /= size
    return y


def normGradient(f, x):
    return normalize(gradient(f, x))
