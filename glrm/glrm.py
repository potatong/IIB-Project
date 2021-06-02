from glrm.convergence import Convergence
from numpy import sqrt, repeat, tile, hstack, array, zeros, ones, sqrt, diag, asarray, hstack, vstack, split, cumsum, reciprocal, square
from numpy.random import randn
import numpy as np
from copy import copy, deepcopy
from numpy.linalg import svd
import cvxpy as cp
import math

# XXX does not support splitting over samples yet (only over features to
# accommodate arbitrary losses by column).

class GLRM(object):

    def __init__(self, A, loss, regX, regY, k, missing_list = None, converge = None, scale=True, obj_type=0,
                 decoder=None, missing_val=None):
        
        self.scale = scale
        # Turn everything in to lists / convert to correct dimensions
        if not isinstance(A, list): A = [A]
        if not isinstance(loss, list): loss = [loss]
        if not isinstance(regY, list): regY = [regY]
        if len(regY) == 1 and len(regY) < len(loss): 
            regY = [copy(regY[0]) for _ in range(len(loss))]
        if missing_list and not isinstance(missing_list[0], list): missing_list = [missing_list]

        if decoder is None:
            loss = [L(Aj) for Aj, L in zip(A, loss)]
        else:
            loss = [L(Aj, d) for Aj, L, d in zip(A, loss, decoder)]

        # save necessary info
        self.A, self.k, self.L = A, k, loss
        if converge == None: self.converge = Convergence()
        else: self.converge = converge

        # initialize cvxpy problems
        self._initialize_probs(A, k, missing_list, regX, regY, obj_type, missing_val)
       
        
    def factors(self):
        # return X, Y as matrices (not lists of sub matrices)
        return self.X, hstack(self.Y)

    def convergence(self):
        # convergence information for alternating minimization algorithm
        return self.converge

    def predict(self):
        # return decode(XY), low-rank approximation of A
        return hstack([L.decode(self.X.dot(yj)) for Aj, yj, L in zip(self.A, self.Y, self.L)])

    def fit(self, max_iters=100, solver=cp.SCS, eps=1e-2, use_indirect=False, warm_start=False):
        
        Xv, Yp, pX = self.probX
        Xp, Yv, pY = self.probY
        self.converge.reset()

        prev_best_obj = None
        prev_best_Xv = None
        prev_best_Yv = None

        i = 0
        # alternating minimization
        while not self.converge.d():
            print('---------- ITERATION {} ----------'.format(i))
            try:
                objX = pX.solve(solver=solver, eps=eps, max_iters=max_iters,
                        use_indirect=use_indirect, warm_start=warm_start)
                Xp.value[:,:-1] = copy(Xv.value)
                print('Residual norm for X minimization: {}'.format(pX.value))
                if pX.value is None or Xv.value is None:
                    raise cp.error.ParameterError

                # can parallelize this
                for ypj, yvj, pyj in zip(Yp, Yv, pY):
                    objY = pyj.solve(solver=solver, eps=eps, max_iters=max_iters,
                            use_indirect=use_indirect, warm_start=warm_start)
                    ypj.value = copy(yvj.value)
                    print('Residual norm for Y minimization: {}'.format(pyj.value))
                    if pyj.value is None or yvj.value is None:
                        raise cp.error.ParameterError

                if self.converge.max_buffer is not None:
                    if prev_best_obj is None:
                        prev_best_obj = objX
                        prev_best_Xv = deepcopy(Xv)
                        prev_best_Yv = deepcopy(Yv)
                    elif objX > prev_best_obj:
                        self.converge.buffer += 1
                    elif objX <= prev_best_obj:
                        prev_best_obj = objX
                        prev_best_Xv = deepcopy(Xv)
                        prev_best_Yv = deepcopy(Yv)
                        self.converge.buffer = 0

                self.converge.obj.append(objX)
                i += 1
            except cp.error.ParameterError:
                print('Error in optimisation, returning previous best solution')
                break
        """
        print(prev_best_obj)
        print(prev_best_Xv.value)
        print([yv.value for yv in prev_best_Yv])
        """

        if self.converge.max_buffer is None:
            self._finalize_XY(Xv, Yv)
        else:
            self._finalize_XY(prev_best_Xv, prev_best_Yv)

        return self.X, self.Y

    def _initialize_probs(self, A, k, missing_list, regX, regY, obj_type, missing_val):
        
        # useful parameters
        m = A[0].shape[0]
        ns = [a.shape[1] for a in A]
        if missing_list == None: missing_list = [[]]*len(self.L)

        # initialize A, X, Y
        B = self._initialize_A(A, missing_list, missing_val)
        X0, Y0 = self._initialize_XY(B, k, missing_list)
        self.X0, self.Y0 = X0, Y0

        # cvxpy problems
        Xv, Yp = cp.Variable((m, k)), [cp.Parameter((k + 1, ns[i]), name='Yp'+str(i), value=copy(Y0[i])) for i in range(len(ns))]
        Xp, Yv = cp.Parameter((m,k+1), name='Xp', value=copy(X0)), [cp.Variable((k+1,ni)) for ni in ns]

        onesM = cp.Constant(ones((m,1)))

        # Get loss function standardization masks
        loss_masks = []
        for mask in self.masks:
            loss_mask = np.copy(mask)
            non_zero = loss_mask != 0
            loss_mask[non_zero] = reciprocal(square(loss_mask[non_zero]))
            loss_masks.append(loss_mask)
        self.loss_masks = loss_masks

        if obj_type == 0:
            obj = sum(L(Aj, cp.multiply(mask, Xv*yj[:-1,:] \
                    + onesM*yj[-1:,:]) + offset) + ry(yj[:-1,:])\
                    for L, Aj, yj, mask, offset, ry in \
                    zip(self.L, A, Yp, self.masks, self.offsets, regY)) + regX(Xv)
            pX = cp.Problem(cp.Minimize(obj))
            pY = [cp.Problem(cp.Minimize(\
                    L(Aj, cp.multiply(mask, Xp*yj) + offset) \
                    + ry(yj[:-1,:]) + regX(Xp))) \
                    for L, Aj, yj, mask, offset, ry in zip(self.L, A, Yv, self.masks, self.offsets, regY)]
        elif obj_type == 1:
            """
            obj = sum(sum(L(Aj[:,i], cp.multiply(mask[:,i], Xv*yj[:-1,i] + onesM*yj[-1:,i]) + offset[:,i]) / mask[0, i] \
                          for i in range(Aj.shape[1])) + ry(yj[:-1,:]) \
                    for L, Aj, yj, mask, offset, ry in \
                    zip(self.L, A, Yp, self.masks, self.offsets, regY)) + regX(Xv)
            pX = cp.Problem(cp.Minimize(obj))
            pY = [cp.Problem(cp.Minimize(\
                    sum(L(Aj[:,i], cp.multiply(mask[:,i], Xp*yj[:,i]) + offset[:,i]) / mask[0, i] \
                        for i in range(Aj.shape[1])) \
                    + ry(yj[:-1,:]) + regX(Xp))) \
                    for L, Aj, yj, mask, offset, ry in zip(self.L, A, Yv, self.masks, self.offsets, regY)]
            """
            # I replaced mask[0] with reciprocal(mask)
            obj = sum(L(Aj, cp.multiply(mask, Xv*yj[:-1,:] \
                    + onesM*yj[-1:,:]) + offset, mask=loss_mask) + ry(yj[:-1,:])\
                    for L, Aj, yj, mask, loss_mask, offset, ry in \
                    zip(self.L, A, Yp, self.masks, self.loss_masks, self.offsets, regY)) + regX(Xv)
            pX = cp.Problem(cp.Minimize(obj))
            pY = [cp.Problem(cp.Minimize(\
                    L(Aj, cp.multiply(mask, Xp*yj) + offset, mask=loss_mask) \
                    + ry(yj[:-1,:]) + regX(Xp))) \
                    for L, Aj, yj, mask, loss_mask, offset, ry in zip(self.L, A, Yv, self.masks, self.loss_masks, self.offsets, regY)]

        self.probX = (Xv, Yp, pX)
        self.probY = (Xp, Yv, pY)

    def _initialize_A(self, A, missing_list, missing_val):
        """ Subtract out means of non-missing, standardize by std. """
        m = A[0].shape[0]
        ns = [a.shape[1] for a in A]
        mean, stdev = [zeros(ni) for ni in ns], [zeros(ni) for ni in ns]
        B, masks, offsets = [], [], []

        # compute stdev for entries that are not missing
        for ni, sv, mu, ai, missing, L in zip(ns, stdev, mean, A, missing_list, self.L):
            
            # collect non-missing terms
            for j in range(ni):
                if missing_val is None:
                    elems = array([ai[i,j] for i in range(m) if (i,j) not in missing])
                else:
                    elems = array([ai[i, j] for i in range(m) if ai[i, j] != missing_val])
                alpha = cp.Variable()
                # calculate standarized energy per column
                #sv[j] = cp.Problem(cp.Minimize(\
                        #L(elems, alpha*ones(elems.shape)))).solve()/len(elems)
                sv[j] = math.sqrt(cp.Problem(cp.Minimize(\
                        L(elems, alpha*ones(elems.shape)))).solve()/len(elems))
                mu[j] = alpha.value

            # Create matrix the same size as A submatrix, with each element equal to mean or stddev of the column
            offset, mask = tile(mu, (m,1)), tile(sv, (m,1))
            mask[mask == 0] = 1
            bi = (ai-offset)/mask # standardize

            # zero-out missing entries (for XY initialization)
            for (i,j) in missing: bi[i,j], mask[i,j] = 0, 0
             
            B.append(bi) # save
            masks.append(mask)
            offsets.append(offset)
        self.masks = masks
        self.offsets = offsets
        return B

    def _initialize_XY(self, B, k, missing_list):
        """ Scale by ration of non-missing, SVD, append col of ones, add noise. """
        A = hstack(bi for bi in B)
        m, n = A.shape

        # normalize entries that are missing
        if self.scale: stdev = A.std(0)
        else: stdev = ones(n)
        mu = A.mean(0)
        C = sqrt(1e-2/k) # XXX may need to be adjusted for larger problems
        A = (A-mu)/stdev + C*randn(m,n)

        # SVD to get initial point
        u, s, v = svd(A, full_matrices = False)
        u, s, v = u[:,:k], diag(sqrt(s[:k])), v[:k,:]
        X0, Y0 = asarray(u.dot(s)), asarray(s.dot(v))*asarray(stdev)

        # append col of ones to X, row of zeros to Y
        X0 = hstack((X0, ones((m,1)))) + C*randn(m,k+1)
        Y0 = vstack((Y0, mu)) + C*randn(k+1,n)

        # split Y0
        ns = cumsum([bj.shape[1] for bj in B])
        if len(ns) == 1: Y0 = [Y0]
        else: Y0 = split(Y0, ns, 1)
        
        return X0, Y0

    def _finalize_XY(self, Xv, Yv):
        """ Multiply by std, offset by mean """
        m, k = Xv.shape
        self.X = asarray(hstack((Xv.value, ones((m,1)))))
        self.Y = [asarray(yj.value)*tile(mask[0,:],(k+1,1)) \
                for yj, mask in zip(Yv, self.masks)]
        for offset, Y in zip(self.offsets, self.Y): Y[-1,:] += offset[0,:]
            
