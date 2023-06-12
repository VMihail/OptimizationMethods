import numpy as np

from lab2.mathUtilityPac.mathUtility import gradient, hessianMatrix, norm, normGradient, normalize
from algorithm import goldenRatioMethod, fibonacciMethod

limited = 50000


def isMonotonic(f, arg, grad, step):
    return f(*(arg - step * grad)) <= f(*arg)


def reduce(f, arg, grad, step):
    while not isMonotonic(f, arg, grad, step):
        step /= 2
    return step


class LambdaScalar:
    def __init__(self, step, scalar):
        self.step = step
        self.scalar = scalar

    def __call__(self, f, x, grad, **kwargs):
        step = reduce(f, x, grad, self.step)
        self.step *= self.scalar
        return step

    def __str__(self):
        return f"{self.step} {self.scalar}"


def speedyDescent(f, x, grad, linearMinimizer):
    res = linearMinimizer(lambda t: f(*(x - t * grad)), 0., 1000., 0.0001)
    return sum(res[-1]) / 2


def constLambda(step):
    return lambda f, x, grad, **kwargs: reduce(f, x, grad, step)


def lambdaScalar(step, scalar):
    return LambdaScalar(step, scalar)


def speedyDescentGolden(f, x, grad):
    return speedyDescent(f, x, grad, goldenRatioMethod)


def speedyDescentFibonacci(f, x, grad):
    return speedyDescent(f, x, grad, fibonacciMethod)


def gradientDescentConst(f, x, eps, t, alpha=1.):
    trajectory = list()
    trajectory.append(x)
    xKx = x
    tK = t
    for k in range(limited):
        xK = xKx
        grad = gradient(f, xK)
        if norm(grad) < eps:
            break
        grad = normalize(grad)
        tK *= alpha
        if k % 30 == 0:
            tK = t
        xKx = xK - tK * grad
        while f(*xKx) - f(*xK) >= 0:
            tK /= 2
            xKx = xK - tK * grad
        trajectory.append(xKx)
        if norm(xKx - xK) < eps and abs(f(*xK) - f(*xKx)) < eps:
            break
    return trajectory


def gradientDescentConstQuickGolden(f, x0, eps):
    trajectory = [x0]
    x_k = x0
    x_kx1 = x0
    for k in range(limited):
        x_k = x_kx1
        grad = gradient(f, x_k)
        if norm(grad) < eps:
            break
        grad = normalize(grad)
        u = goldenRatioMethod(lambda t: f(*(x_k - t * grad)), 0., 1000., eps)
        t_k = sum(u[-1]) / 2
        x_kx1 = x_k - t_k * grad
        trajectory.append(x_kx1)
        length = norm(x_kx1 - x_k)
        if length < eps and abs(f(*x_k) - f(*x_kx1)) < eps:
            break
    return trajectory


def gradientMethodQuickFibonacci(f, x0, eps):
    trajectory = [x0]
    x_k = x0
    x_kx1 = x0
    k = 0
    while k < limited:
        k += 1
        x_k = x_kx1
        grad = gradient(f, x_k)
        if norm(grad) < eps:
            break
        grad = normalize(grad)
        f_min = lambda t: f(*(x_k - t * grad))
        u = fibonacciMethod(f_min, 0., 1000., eps)
        t_k = sum(u[-1]) / 2
        x_kx1 = x_k - t_k * grad
        trajectory.append(x_kx1)
        length = norm(x_kx1 - x_k)
        if length < eps and abs(f(*x_k) - f(*x_kx1)) < eps:
            break
    return trajectory


def conjugateMethod(f, x, eps, funcRight):
    trajectory = [x]
    k = 0
    xK = x
    xNext = x
    grad = normGradient(f, xK)
    d = -normGradient(f, xK)
    while k < limited:
        xK = xNext
        grad_prev = grad
        grad = gradient(f, xK)
        if norm(grad) < eps:
            break
        grad = normalize(grad)
        k += 1
        b = funcRight(grad, grad_prev)
        d = -grad + b * d
        f_min = lambda ttt: f(*(xK + ttt * d))
        u = goldenRatioMethod(f_min, 0., 10000000., eps)
        t = sum(u[-1]) / 2
        xNext = xK + t * d
        trajectory.append(xNext)
        if norm(xNext - xK) < eps and abs(f(*xNext) - f(*xK)) < eps:
            break
    return trajectory


def conjugateGradientDescent(f, x, eps):
    hessian = hessianMatrix(f, x)
    if not np.all(np.linalg.eigvals(hessian) > 0):
        return None
    return conjugateMethod(f, x, eps, lambda grad, grad_prev: norm(grad) ** 2 / norm(grad_prev) ** 2)


def conjugateDirectionMethod(f, x, eps):
    return conjugateMethod(f, x, eps, lambda grad, grad_prev: np.dot(grad, grad - grad_prev) / norm(grad) ** 2)


def newtonMethod(f, x, eps):
    trajectory = [x]
    k = 0
    x = np.array(x)
    prevX = x
    grad = gradient(f, x)
    while (norm(grad) >= eps and k < limited and prevX is x or (
                    norm(prevX - x) >= eps and abs(f(*x) - f(*prevX)) >= eps)):
        hessian = np.array(hessianMatrix(f, x))
        try:
            inv_hessian = np.linalg.inv(hessian)
        except np.linalg.LinAlgError:
            return None
        else:
            if np.linalg.det(inv_hessian) > 0:
                direction = -np.matmul(inv_hessian, np.atleast_2d(grad).transpose())
                direction = direction.transpose()[0]
            else:
                direction = -grad
        prevX = x.copy()
        x += direction
        grad = gradient(f, x)
        k += 1
        trajectory.append(x)
    return trajectory
