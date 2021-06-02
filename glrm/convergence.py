from matplotlib import pyplot as plt

class Convergence(object):

    def __init__(self, TOL = 1e-3, max_iters = 1e3, max_buffer = None):
        self.TOL = TOL
        self.max_iters = max_iters
        self.max_buffer = max_buffer # number of iterations to wait for a lower objective than previous best
        self.reset()

    def reset(self):
        self.obj = []
        self.val = []
        self.buffer = 0

    def d(self): # if converge.d == True:
        # return True if converged
        if len(self) < 2: return False
        if len(self) > self.max_iters: 
            print("hit max iters for convergence object")
            return True
        if self.max_buffer is not None:
            if self.buffer == self.max_buffer:
                return True
        return abs(self.obj[-1] - self.obj[-2])/self.obj[-2] < self.TOL

    def __len__(self):
        return len(self.obj)

    def __str__(self):
        return str(self.obj)

    def __repr__(self):
        return str(self.obj)

    def plot(self):
        plt.plot(self.obj)
        plt.title("model error")
        plt.xlabel("iteration")
        plt.show()
