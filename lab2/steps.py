from alg import goldenRatioMethod, fibonacciMethod


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
