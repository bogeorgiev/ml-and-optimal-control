import numpy as np
from numpy.linalg import inv
from scipy.optimize import approx_fprime


class LBFGSB:

    _delta = 1e-6

    def __init__(self, func, x0, l=None, u=None, m=20, tol=1e-5, max_iter=100, history=True, logging=False):
        # x0, l, u = self.format_inputs(func, x0, l, u, m, tol, max_iter, history, logging)
        self.func = func
        self.x0 = x0
        self.l = l
        self.u = u
        self.m = m
        self.tol = tol
        self.max_iter = max_iter
        self.history = history


    def format_inputs(self, func, x0, l, u, m, tol, max_iter, history, logging):
        assert callable(func), "func is of type {}, must be <class 'function'>".format(type(func))
        assert isinstance(x0,list), "\'x0\' is of type {}, must me {}".format(type(x0), type([]))
        x0 = np.array(x0).reshape((len(x0),))
        if l:
            assert isinstance(l,list), "\'l\' is of type {}, must me {}".format(type(l), type([]))
            assert len(l)==len(x0), "length of \'l\' must be equal to length of \'x0\'"
        elif l==None:
            l = np.ones_like(x0)*np.finfo(float).min
        else:
            l = np.array(l).reshape((len(l),))
        if u:
            assert isinstance(u,list), "\'u\' is of type {}, must me {}".format(type(u), type([]))
            assert len(u)==len(x0), "length of \'u\' must be equal to length of \'x0\'"
        elif u==None:
            u = np.ones_like(x0)*np.finfo(float).max
        else:
            u = np.array(u).reshape((len(lu),))
        for i in range(len(x0)):
            if l[i]==None:
                l[i] = np.finfo(float).min
            if u[i]==None:
                u[i] = np.finfo(float).max
        assert isinstance(m,int), "\'m\' must be integer, but it is {}".format(type(m))
        assert isinstance(max_iter,int), "\'max_iter\' must be integer, but it is {}".format(type(max_iter))
        assert isinstance(tol,float), "\'tol\' must be float, but it is {}".format(type(tol))
        assert isinstance(history,bool), "\'history\' must be bool, but it is {}".format(type(history))
        assert isinstance(logging,bool), "\'history\' must be bool, but it is {}".format(type(logging))
        return x0, l, u

    def solve(self):

        n = len(self.x0)
        Y = np.array([], dtype=np.float64).reshape((n,0))
        S = np.array([], dtype=np.float64).reshape((n,0))
        W = np.zeros((n,1), dtype=np.float64)
        M = np.zeros((1,1), dtype=np.float64)
        theta = 1

        l = self.l
        u = self.u
        x = self.x0
        f = self.func(x)
        g = approx_fprime(x, self.func, self._delta)

        k = 0

        while k < self.max_iter and self.optimal_tol(x,g,l,u) > self.tol:
            print(f, k)
            xk = x.copy()
            gk = g.copy()

            xc, c = self.cauchy_point(xk,gk,l,u,theta,W,M)
            xbar, line_search_f = self.direct_primal_method(xk,gk,l,u,xc,c,theta,W,M)
            alpha = 1.0
            if line_search_f:
                dk = xbar - xk
                alpha = self.wolfe_condition(self.func, xk, f, gk, dk)
                x = xk + alpha*dk
                f = self.func(x)
                g = approx_fprime(x, self.func, self._delta)
                y = g - gk
                s = x - xk
                curv = np.abs(np.dot(s,y))

                if curv <= np.finfo(float).eps:
                    print(' Warning: negative curvature detected\n')
                    print('          skipping LBFGS update\n')
                    k = k+1
                    continue

                if k < self.m:
                    Y = np.concatenate((Y, y.reshape((n,1))), axis=1)
                    S = np.concatenate((S, s.reshape((n,1))), axis=1)
                else:
                    Y[:,:self.m-1] = Y[:,1:]
                    S[:,:self.m-1] = S[:,1:]
                    Y[:,-1] = y
                    S[:,-1] = s

                theta = (np.dot(y,y))/(np.dot(y,s))
                W = np.concatenate((Y, theta*S), axis=1)
                A = np.dot(S.T,Y)
                L = np.tril(A,-1)
                D = -np.diag(np.diag(A))
                MM = np.concatenate((np.concatenate((D,L.T),axis=1), np.concatenate((L,np.dot(theta*S.T,S)),axis=1)))
                M = np.linalg.inv(MM)

                k = k+1

                if k == self.max_iter:
                    print(' Warning: maximum number of iterations reached!\n')

                if self.optimal_tol(x,g,l,u) < self.tol:
                    print(' Stopping because convergence tolerance met!\n')

        return x


    def optimal_tol(self, x, g, l, u):

        proj_grad = x-g

        for i in range(len(proj_grad)):
            if proj_grad[i] < l[i]:
                proj_grad[i] = l[i]
            elif proj_grad[i] > u[i]:
                proj_grad[i] = u[i]

        return max(abs(proj_grad - x))


    def breakpoints(self, x,g,l,u):

        n = len(x)
        t = np.zeros(n)
        d = -g
        for i in range(n):
            if g[i] < 0:
                t[i] = (x[i] - u[i])/g[i]
            elif g[i] > 0:
                t[i] = (x[i] - l[i])/g[i]
            else:
                t[i] = np.finfo(float).max
            if t[i] == 0.0:
                d[i] = 0.0

        return t,d

    def sorted_indices(self, t):

        n = len(t)
        ind = np.linspace(0,n-1,n)
        F = np.concatenate((t,ind)).reshape((2,n)).T
        F = F[F[:,0].argsort(),][:,1]

        return F.astype(int)


    def cauchy_point(self, x,g,l,u,theta,W,M):

        t,d = self.breakpoints(x,g,l,u)
        # initialize
        F = self.sorted_indices(t)
        p = np.dot(d,W)
        c = 0
        fprime = -np.dot(d,d)
        fsecond = -theta*fprime - np.dot(p,np.dot(M,p))
        dtmin = -fprime/fsecond
        told = 0
        for i in range(len(F)):
            if t[F[i]] > 0:
                break
        b = F[i]
        tt = t[b]
        dt = tt - told

        xc = x.copy()
        while dtmin > dt and i <len(x):
            if d[b]>0:
                xc[b] = u[b]
            elif d[b] < 0:
                xc[b] = l[b]
            zb = xc[b] - x[b]
            c = c + dt*p
            wbt = W[b,:]
            fprime = fprime + dt*fprime + g[b]*g[b] + theta*g[b]*zb - g[b]*np.dot(wbt,np.dot(M,c))
            fsecond = fsecond -theta*g[b]*g[b] - 2*g[b]*np.dot(wbt,np.dot(M,p)) - g[b]*g[b]*np.dot(wbt,np.dot(wbt,M))
            p = p + g[b]*wbt.T
            d[b] = 0
            dtmin = -fprime/fsecond
            told = tt
            i+=1
            if i < len(x):
                b = F[i]
                tt = t[b]
                dt = tt - told

        dtmin = max(dtmin,0)
        told = told + dtmin
        for j in range(i,len(xc)):
            indx = F[j]
            xc[indx] = x[indx] + told*d[indx]
        c = c + dtmin*p

        return xc, c


    def alpha_star(self, l,u,xc,du,free_ind):

        alpha_star = 1
        for i in range(len(free_ind)):
            ind = free_ind[i]
            if du[i] > 0:
                alpha_star = min(alpha_star, (u[ind]-xc[ind])/du[i] )
            else:
                alpha_star = min(alpha_star, (l[ind]-xc[ind])/du[i] )

        return alpha_star


    def direct_primal_method(self, x, g, l, u, xc, c, theta, W, M):

        line_search_f = True
        n = len(x)
        free_ind = []
        for i in range(len(xc)):
            if xc[i] != u[i] and xc[i] != l[i]:
                free_ind.append(i)

        num_free_ind = len(free_ind)
        Z = np.eye(n)[:,free_ind]

        if num_free_ind == 0:
            return xc, False

        WTZ = np.dot(W.T,Z)
        r = g + theta*(xc-x) - (np.dot(W,np.dot(M,c)))
        rc = np.zeros((num_free_ind,1))
        for i in range(num_free_ind):
            rc[i] = r[free_ind[i]]

        invtheta = 1/theta

        v = np.dot(M,np.dot(WTZ,rc))
        N = invtheta*np.dot(WTZ,WTZ.T)
        N = np.eye(len(N)) - np.dot(M,N)
        v = np.linalg.lstsq(N, v)[0]
        du = -invtheta*rc - invtheta**2 * np.dot(WTZ.T,v)

        alpha_star = self.alpha_star(l,u,xc,du,free_ind)
        dstar = alpha_star*du
        xbar = xc.copy()
        for i in range(num_free_ind):
            ind = free_ind[i]
            xbar[ind] = xbar[ind] + dstar[i]

        return xbar, line_search_f


    def wolfe_condition(self, func,x0,f0,g0,p):

        delta = 1e-6
        c1 = 1e-4
        c2 = 0.9
        alpha_max = 2.5
        alpha_im1 = 0
        alpha_i = 1
        f_im1 = f0
        dphi0 = np.dot(g0,p)
        i = 0
        max_iters = 20

        while True:
            x = x0 + alpha_i*p
            f_i = func(x)
            g_i = approx_fprime(x, func, delta)
            if f_i > f0 + c1*dphi0 or (i > 1 and f_i >= f_im1):
                alpha = self.alpha_zoom(func,x0,f0,g0,p,alpha_im1,alpha_i)
                break
            dphi = np.dot(g_i,p)
            if abs(dphi) <= -c2*dphi0:
                alpha = alpha_i;
                break
            if dphi >= 0:
                alpha = self.alpha_zoom(func,x0,f0,g0,p,alpha_i,alpha_im1)
                break

            alpha_im1 = alpha_i
            f_im1 = f_i
            alpha_i = alpha_i + 0.8*(alpha_max-alpha_i)

            if i > max_iters:
                alpha = alpha_i
                break
            i = i+1

        return alpha


    def alpha_zoom(self, func,x0,f0,g0,p,alpha_lo,alpha_hi):
        delta = 1e-6
        c1 = 1e-4
        c2 = 0.9
        i = 0
        max_iters = 20
        dphi0 = np.dot(g0,p)

        while True:
            alpha_i = 0.5*(alpha_lo + alpha_hi)
            alpha = alpha_i
            x = x0 + alpha_i*p
            f_i = func(x)
            g_i = approx_fprime(x, func, delta)
            x_lo = x0 + alpha_lo*p
            f_lo = func(x_lo)
            if (f_i > f0 + c1*alpha_i*dphi0) or ( f_i >= f_lo):
                alpha_hi = alpha_i
            else:
                dphi =np.dot(g_i,p)
                if abs(dphi) <= -c2*dphi0:
                    alpha = alpha_i
                    break
                if dphi * (alpha_hi-alpha_lo) >= 0:
                    alpha_hi = alpha_lo
                alpha_lo = alpha_i
            i = i+1
            if i > max_iters:
                alpha = alpha_i;
                break

        return alpha
