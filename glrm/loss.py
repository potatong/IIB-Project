import cvxpy as cp
from numpy import ones, maximum, minimum, sign, floor, ceil, cumsum, argmax
import numpy as np

"""
Abstract loss class and canonical loss functions.
"""

# Abstract Loss class
class Loss(object):
    def __init__(self, A, decoder=None): return
    def loss(self, A, U): raise NotImplementedError("Override me!")
    def encode(self, A): return A # default
    def decode(self, A): return A # default
    def __str__(self): return "GLRM Loss: override me!"
    def __call__(self, A, U, mask=None): return self.loss(A, U, mask)

# Canonical loss functions
class QuadraticLoss(Loss):
    def loss(self, A, U, mask):
        x = cp.Constant(A) - U
        if mask is None:
            return cp.sum_squares(x)
            """
            if x.ndim == 2:
                return cp.norm(cp.Constant(A) - U, "fro")
            else:
                return cp.norm(cp.Constant(A) - U, 2)
            """
        else:
            #result = sum((cp.norm(x[:,i], 2) * mask[0][i]) for i in range(len(mask[0])))
            result = cp.sum(cp.multiply(cp.square(x), cp.Constant(mask)))
            return result

    def __str__(self): return "quadratic loss"

class HuberLoss(Loss):
    a = 1.0 # XXX does the value of 'a' propagate if we update it?
    def loss(self, A, U, mask):
        if mask is None:
            return cp.sum(cp.huber(cp.Constant(A) - U, self.a))
        else:
            #x = cp.Constant(A) - U
            #result = sum((cp.sum(cp.huber(x[:,i], self.a)) / mask[i]) for i in range(len(mask)))
            result = cp.sum(cp.multiply(cp.huber(cp.Constant(A) - U, self.a), cp.Constant(mask)))
            return result
    def __str__(self): return "huber loss"

# class FractionalLoss(Loss):
#     PRECISION = 1e-10
#     def loss(self, A, U):
#         B = cp.Constant(A)
#         U = cp.maximum(U, self.PRECISION) # to avoid dividing by zero
#         return cp.maximum(cp.multiply(cp.inv_pos(cp.pos(U)), B-U), \
#         return maximum((A - U)/U, (U - A)/A)
# 

class HingeLoss(Loss):
    def loss(self, A, U, mask):
        if mask is None:
            return cp.sum(cp.pos(ones(A.shape)-cp.multiply(cp.Constant(A), U)))
        else:
            #return sum(cp.sum(cp.pos(ones(A.shape) - cp.multiply(cp.Constant(A), U)), axis=0)[i] * cp.Constant(mask[0][i]) \
                       #for i in range(len(mask[0])))
            return cp.sum(cp.multiply(cp.pos(ones(A.shape) - cp.multiply(cp.Constant(A), U)), cp.Constant(mask))) # Faster but worse result

    def decode(self, A): return sign(A) # return back to Boolean
    def __str__(self): return "hinge loss"


class CategoricalLoss(HingeLoss):
    def __init__(self, A, decoder):
        self.decoder = decoder
    def decode(self, A):
        decoded_A = np.full((A.shape[0], A.shape[1]), -1)
        col_nos = list(cumsum(self.decoder))
        col_nos.insert(0, 0)
        for i in range(len(col_nos) - 1):
            block = A[:, col_nos[i]:col_nos[i+1]]
            for r in range(len(block)):
                max_i = argmax(block[r])
                decoded_A[r, col_nos[i]+max_i] = 1

        return decoded_A # return back to Categorical
    def __str__(self): return "categorical loss"


class OrdinalLoss(Loss):
    def __init__(self, A):
        self.Amax, self.Amin = A.max(), A.min()

    def loss(self, A, U, mask):
        if mask is None:
            return cp.sum(sum(cp.multiply(1*(b >= A),\
                    cp.pos(U-b*ones(A.shape))) + cp.multiply(1*(b < A), \
                    cp.pos(-U + (b+1)*ones(A.shape))) for b in range(int(self.Amin), int(self.Amax))))
        else:
            """
            return sum(cp.sum(sum(cp.multiply(1*(b >= A),\
                    cp.pos(U-b*ones(A.shape))) + cp.multiply(1*(b < A), \
                    cp.pos(-U + (b+1)*ones(A.shape))) for b in range(int(self.Amin), int(self.Amax))), axis=0)[i] / mask[i] \
                       for i in range(len(mask)))
            """
            return cp.sum(cp.multiply(sum(cp.multiply(1 * (b >= A), \
                                          cp.pos(U - b * ones(A.shape))) + cp.multiply(1 * (b < A), \
                                                                                       cp.pos(-U + (b + 1) * ones(
                                                                                           A.shape))) for b in
                              range(int(self.Amin), int(self.Amax))), cp.Constant(mask)))

    def decode(self, A): return maximum(minimum(A.round(), self.Amax), self.Amin)
    def __str__(self): return "ordinal loss"
