import numpy as np
from numpy.linalg import norm
from numpy.linalg import inv
from numpy.linalg import lstsq
from scipy.optimize import approx_fprime
from scipy.sparse.linalg import cg

from scipy.optimize import line_search
np.seterr(over="raise")

import time
import pdb

def get_optimality(x,g,l,u):

    projected_g = x-g
    projected_g[projected_g < l] = l[projected_g < l]
    projected_g[projected_g > u] = u[projected_g > u]
            
    projected_g = projected_g - x

    return max(abs(projected_g))


def get_breakpoints(x,g,l,u):

    t = np.zeros(len(x))
    t[g<0] = ( x[g<0] - u[g<0] )/g[g<0]
    t[g>0] = ( x[g>0] - l[g>0] )/g[g>0]
    t[g==0] = np.finfo(float).max

    return t[t!=0]


def get_cauchy_point(x,g,l,u,theta,W,M):
    tt = get_breakpoints(x,g,l,u)
    d = -g
    xc = x.copy()
    p = np.dot(d, W)
    c = 0
    fp = -np.dot(d,d)
    fpp = -theta*fp - np.dot(np.dot(M,p),p)
    dt_min = - fp/fpp
    t_old = 0
    F = tt.argsort()
    b = F[0]
    t = tt[b]
    dt = t - t_old
    i = 0
    while dt_min > dt and i < len(tt):
        if d[b]>0:
            xc[b] = u[b]
        elif d[b] < 0:
            xc[b] = l[b]
        zb = xc[b] - x[b]
        c = c + dt*p
        gb = g[b]
        wbt = W[b,:]
        fp = fp + dt*fp + gb*gb + theta*gb*zb - gb*np.dot(wbt,np.dot(M,c))
        fpp = fpp -theta*gb*gb - 2*gb*np.dot(wbt,np.dot(M,p)) - gb*gb*np.dot(wbt,np.dot(wbt, M))
        p = p + gb*wbt
        d[b] = 0.
        try:
            dt_min = -fp/fpp
        except:
            dt_min = 0
        t_old = t
        i+=1
        if i < len(tt):
            b = F[i]
            t = tt[F[i]]
            dt = t - t_old
    dt_min = max(dt_min, 0)
    t_old = t_old + dt_min
    inds = F[i:]
    xc[inds] = x[inds] + t_old*d[inds]
    c = c + dt_min*p
    return xc, c


def find_alpha(l,u,xc,du,free_vars_idx):

    alpha_star = 1.
    for i,idx in enumerate(free_vars_idx):
        if du[i] > 0:
            alpha_star = min(alpha_star, (u[idx]-xc[idx])/du[i] )
        elif du[i] < 0:
            alpha_star = min(alpha_star, (l[idx]-xc[idx])/du[i] )

    return alpha_star


def subspace_min(x,g,l,u,xc,c,theta,W,M):
    
    n = len(x)
    free_vars_idx = np.argwhere((xc != u) & (xc != l))[:,0]
    Z  = np.eye(n)[:,free_vars_idx]

    if len(free_vars_idx) == 0:
        return xc, False

    WTZ = np.dot(W.T,Z)
    rr = g + theta*(xc - x) - np.dot(W, np.dot(M,c))
    r = np.zeros((n,1))
    r = rr[free_vars_idx]
    invtheta = 1.0/theta
    v = np.dot(M, np.dot(WTZ,r))
    N = invtheta * np.dot(WTZ, WTZ.T)
    v = lstsq(N, v, rcond=None)[0]
#     v = np.dot(inv(N), v)
    du = -invtheta*r - invtheta * invtheta * np.dot(v, WTZ)
    alpha_star = find_alpha(l,u,xc,du,free_vars_idx)
    d_star = alpha_star*du
    # xbar = xc.copy()
    for i,idx in enumerate(free_vars_idx):
        xc[idx] = xc[idx] + d_star[i]
    # return xbar, True
    return xc, True

def subspace_min1(x,g,l,u,xc,c,theta,W,M):

    free_vars_idx = np.argwhere((xc != u) & (xc != l))[:,0]
    Z  = np.eye(len(x))[:,free_vars_idx]

    if len(free_vars_idx) == 0:
        return xc, False

    Bk = theta*np.eye(len(free_vars_idx)) - np.dot(np.dot(Z.T,W), np.dot(M,np.dot(W.T,Z)))
    
    rr = g + theta*(xc - x) - np.dot(W, np.dot(M,c))
    rr = rr[free_vars_idx]
    p = -rr

    # du = cg(Bk,p)[0]
    # alpha1 = find_alpha(l,u,xc,du,free_vars_idx)

    r2 = np.dot(rr,rr)
    du = np.zeros_like(p)
    norm_rr = norm(rr)
    alpha1 = 0
    while norm_rr > np.max([0.1, np.sqrt(norm_rr)])*norm_rr:
        alpha1 = find_alpha(l,u,xc,du,free_vars_idx)
        q = np.dot(Bk,p)
        alpha2 = r2/np.dot(p.T,q)
        if alpha2>alpha1:
            break
        else:
            du += alpha2*p
            rr += alpha2*q
            r1 = r2
            r2 = np.dot(rr,rr)
            p = -rr  + r2/r1*p

    du += alpha1*p
    xc[free_vars_idx] += du
    return xc, True 


def alpha_zoom(func,x0,f0,g0,p,alpha_lo,alpha_hi, delta=1e-3, c1=1e-4, c2=0.9, max_iters=10):

    i = 0
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
            dphi = np.dot(g_i,p)
            if abs(dphi) <= -c2*dphi0:
                alpha = alpha_i
                break
            if dphi * (alpha_hi-alpha_lo) >= 0:
                alpha_hi = alpha_lo
            alpha_lo = alpha_i
        i = i+1
        if i > max_iters:
            alpha = alpha_i
            break

    return alpha


def strong_wolfe(func,x0,f0,g0,p, delta=1e-8, c1=1e-4, c2=0.9, alpha_max=1.5, alpha_im1=0, alpha_i=1., max_iters=10):

    f_im1 = f0
    dphi0 = np.dot(g0,p)
    i = 0

    while True:
        x = x0 + alpha_i*p
        f_i = func(x)
        g_i = approx_fprime(x, func, delta)
        if f_i > f0 + c1*dphi0 or (i > 1 and f_i >= f_im1):
            alpha = alpha_zoom(func,x0,f0,g0,p,alpha_im1,alpha_i)
            break
        dphi = np.dot(g_i,p)
        if abs(dphi) <= -c2*dphi0:
            alpha = alpha_i
            break
        if dphi >= 0:
            alpha = alpha_zoom(func,x0,f0,g0,p,alpha_i,alpha_im1)
            break

        alpha_im1 = alpha_i
        f_im1 = f_i
        alpha_i = alpha_i + 0.8*(alpha_max-alpha_i)

        if i > max_iters:
            alpha = alpha_i
            break
        i = i+1

    return alpha


def solve(func, x, l, u, m=10, max_iters=20):

    delta = 1e-08
    tol = 1e-05

    n = len(x)
    Y = np.zeros((n,0))
    S = np.zeros((n,0))
    W = np.zeros((n,1))
    M = np.zeros((1,1))
    theta = 1

    f = func(x)
    g = approx_fprime(x, func, delta)
    k = 0
    while get_optimality(x,g,l,u) > tol and k < max_iters:
        print("Iter :", k, "\t Obj fun: ", f)
        g_old = g.copy()

        xc, c = get_cauchy_point(x,g,l,u,theta,W,M)
        xbar, line_search_flag = subspace_min1(x,g,l,u,xc,c,theta,W,M)

        alpha = 1.
        if line_search_flag:
            alpha = strong_wolfe(func,x,f,g,xbar-x)
        s = alpha * (xbar - x)
        x = x + alpha * (xbar - x)  

        # update the LBFGS data structures  
        f = func(x)
        g = approx_fprime(x, func, delta)
        y = g - g_old
        curv = abs(np.dot(s,y))
        if (curv < np.finfo(float).eps*np.dot(y,y)):
            print(' Warning: negative curvature detected\n')
            print('          skipping LBFGS update\n')
            k = k+1
            continue
        if Y.shape[1] < m:
            Y = np.concatenate((Y, y.reshape((n,1))), axis=1)
            S = np.concatenate((S, s.reshape((n,1))), axis=1)
        else:
            Y[:,:m-1] = Y[:,1:]
            S[:,:m-1] = S[:,1:]
            Y[:,-1] = y
            S[:,-1] = s 

        theta = (np.dot(y,y))/(np.dot(y,s))
        W = np.concatenate((Y, theta*S), axis=1)
        A = np.dot(S.T,Y)
        L = np.tril(A,-1)
        D = -np.diag(np.diag(A))
        MM = np.concatenate((np.concatenate((D,L.T),axis=1), np.concatenate((L,np.dot(theta*S.T,S)),axis=1)))
        M = inv(MM)

        k = k+1
        

    if k == max_iters:
        print(' Stopped: maximum number of iterations reached!\n')

    if get_optimality(x,g,l,u) < tol:
        print(' Stopping because convergence tolerance met!\n')
    
    return x
 
