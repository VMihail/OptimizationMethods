from bisect import bisect_left


class Fibonacci:
    def __init__(self):
        self.array = [1, 1]

    def getNext(self):
        next_fib = sum(self.array[-2:])
        self.array.append(next_fib)

    def getFibByNumber(self, n):
        while n >= len(self.array):
            self.getNext()
        return self.array[n]

    def upperBound(self, fib):
        if fib <= self.array[-1]:
            return bisect_left(self.array, fib)
        while fib > self.array[-1]:
            self.getNext()
        return len(self.array) - 1
