#!/usr/bin/python  
# -*- coding: utf-8 -*-

'''
An implementation of Algorithms in numerical optimization

TrustRegion_dogleg:trust region method with subproblems
solved by the Dogleg method

SteepestDescent:Steepest Descent method

ConjugateGradient:Conjugate Gradient Method

Newton: Newton's method

QuasiNewton: quasi-Newton method

PenaltySimple: penalty function method

'''

import tensorflow as tf
import numpy as np
from inspect import isfunction
import copy

def dogleg(p_u, p_b, delta, tau = 2):
    p_u2 = np.linalg.norm(p_u, 2)**2
    p_b2 = np.linalg.norm(p_b, 2)**2
    p_ub2 = np.linalg.norm(p_b-p_u, 2)**2
    if p_u2 > delta**2:
        tau = delta / np.linalg.norm(p_u,2)
    elif (p_b2 > delta**2) and (p_u2 <= delta**2):
        tau = np.sqrt((delta**2 - p_u2)/(p_ub2))+1
    if (tau >= 0)and(tau <= 1):
        p_k = tau * p_u
    elif (tau >= 1)and(tau <= 2):
        p_k = p_u + (tau - 1)*(p_b - p_u)
    return p_k

def getGrad(f, x_value):
    x = tf.placeholder(tf.float32, shape=len(x_value) )
    f_grad = tf.gradients(f(x), x)
    sess = tf.Session()  
    f_g = sess.run(f_grad, feed_dict = {x:x_value})
    f_g_value = f_g[0]
    return f_g_value

def getHess(f, x_value):
    x = tf.placeholder(tf.float32, shape=len(x_value) )  
    f_grad = tf.hessians(f(x), x)
    sess = tf.Session()  
    f_g = sess.run(f_grad, feed_dict = {x:x_value})
    f_g_value = f_g[0]
    return f_g_value

def ExactLineSearch_quadratic(f, x_k, p_k):
    ## you can use this Exact Line search only when f is a quadratic function
    f_grad = getGrad(f, x_k)
    f_hess = getHess(f, x_k)
    f_grad = np.mat(f_grad).T
    f_hess = np.mat(f_hess)
    alpha_k = (p_k.T * (-f_grad)) / (p_k.T * f_hess * p_k)
    return alpha_k


def QuasiNewton(f, *x_0,  HUpdateMethod = 'BFGS', LineSearch = ExactLineSearch_quadratic, tolerance = 0.0001):
    if not hasattr(f, "paraLength"):
        raise Exception("Please make sure that f has f.paraLength,  which is \
              the length of x_0,that is \"f.paraLength = len(x)\" ")
    assert isfunction(f)
    input_size  = f.paraLength
    # initializing x_0
    if len(x_0) != 0:
        x_0 = np.mat(x_0[0]).T
    else:
        x_0 = np.mat(np.zeros((input_size,1))).T
    H_0 = np.mat(np.eye(input_size))
    print('------')
    print(x_0)
    g_0 = getGrad(f, (x_0.T).tolist()[0])
    g_0 = np.mat(g_0).T
    LSfun = LineSearch
    ## H_k update function 
    s_k = tf.placeholder(tf.float32, shape=np.shape(x_0))
    y_k = tf.placeholder(tf.float32, shape=np.shape(x_0))
    H_k = tf.placeholder(tf.float32, shape=(input_size, input_size)) 
    I_mat = np.eye(input_size)
    divider = tf.matmul(tf.transpose(s_k), y_k)
    update_3 = tf.matmul(s_k, tf.transpose(s_k))/divider
    if HUpdateMethod == 'BFGS':     
        update_1 = I_mat - tf.matmul(s_k, tf.transpose(y_k))/divider
        update_2 = I_mat - tf.matmul(y_k, tf.transpose(s_k))/divider
        H_update = tf.matmul(tf.matmul(update_1, H_k), update_2) + update_3
    elif HUpdateMethod == 'DFP':
        update_1 = tf.matmul(tf.matmul(tf.matmul(H_k, y_k), tf.transpose(y_k)), H_k)
        update_2 = tf.matmul(tf.matmul(tf.transpose(y_k), H_k), y_k)
        H_update = H_k + update_3 - update_1/update_2
        
    sess = tf.Session()
    k_num = 1
    while np.linalg.norm(g_0,2) > tolerance: 
        print('The %d iteration...'% (k_num))
        p_0 = - np.dot(H_0, g_0)
        a_0 = LSfun(f, (x_0.T).tolist()[0], p_0)
        x_1 = x_0 + np.dot(p_0, a_0)
        g_1 = np.mat(getGrad(f, (x_1.T).tolist()[0])).T
        s_0 = x_1 - x_0
        y_0 = g_1 - g_0
        H_1 = sess.run(H_update, feed_dict = {s_k:s_0, y_k:y_0, H_k:H_0})
        # for the next iteration
        x_0 = copy.deepcopy(x_1)
        g_0 = copy.deepcopy(g_1)
        H_0 = copy.deepcopy(H_1)
        print('x_%d:'%(k_num))
        print(x_0)
        print('gradient at x_%d = %f'%(k_num, np.linalg.norm(g_0,2)))
        k_num += 1
        
    print('After %d iteration, the minimum point = ' % (k_num-1))
    print(x_0)
    print('And the minimum value of f = ')
    print(f((x_0.T).tolist()[0]))
    return x_0, f((x_0.T).tolist()[0] )

    


def TrustRegion_dogleg(f, delta = 0.5, eta = 0, *x_0, tolerance= 0.0001):
    if not hasattr(f, "paraLength"):
        raise Exception("Please make sure that f has f.paraLength,  which is \
              the length of x_0,that is \"f.paraLength = len(x)\" ")
    assert delta > 0
    assert isfunction(f)
    input_size  = f.paraLength
    # initializing x_0
    if len(x_0) != 0:
        pass
    else:
        x_0 = np.zeros(input_size)*0.05
    delta_hat = 2*delta
    x_k = copy.deepcopy(x_0)
    delta_k = copy.deepcopy(delta)
    g_k = getGrad(f, x_k) 
    k_num = 0
    while np.linalg.norm(g_k, 2) > tolerance:
        print('The %d iteration...' % (k_num+1))
        g_k = getGrad(f, x_k) 
#         print('g_%d:'%(k_num))
#         print(g_k)
        g_k = np.mat(g_k).T
        g_k_square = tf.pow(tf.norm(g_k, 2), 2)
        ##nump array to tf.matrix       
        H_k = np.mat(getHess(f, x_k))
#         print('H_%d:'%(k_num))
#         print(H_k)
        print('H_%d' %(k_num))
        print(H_k)
        denom = tf.matmul(tf.matmul(g_k.T, H_k), g_k)
        p_u = - g_k * g_k_square /denom
        sess = tf.Session()
        p_u = sess.run(p_u)
        p_b = - tf.matmul(tf.matrix_inverse(H_k), g_k)
        sess = tf.Session()
        p_b = sess.run(p_b)
#         print('p_u and p_b')
#         print(p_u)
#         print(p_b)
        p_k = dogleg(p_u, p_b, delta_k)
#         print('p_%d'% (k_num))
#         print(p_k)
        p_mat = np.mat(p_k)
#         print(g_mat.T)
#         print(p_mat)
        pred = tf.matmul(g_k.T, p_mat)[0][0] + \
                1/2 * tf.matmul(tf.matmul(p_mat.T, H_k), p_mat)[0][0]
        #pred = np.dot(g_mat.T, p_mat) + \
                #1/2 * np.dot(np.dot(p_mat.T, H_k), p_mat)
        p_k = p_k.T[0]
        ared = f(x_k + p_k) - f(x_k)
        sess  = tf.Session()
        pred = sess.run(pred)
#         print('pred and ared')
#         print(pred)
#         print(ared)
        rho_k = abs(ared / pred)
#         print('rho_%d'%(k_num))
#         print(rho_k)
        if rho_k < 1/4:
            delta_k = delta_k * 1/4
        else:
            if rho_k > 3/4 and np.linalg.norm(p_k, 2) == delta_k:
                delta_k = min(2*delta_k, delta_hat)
        if rho_k > eta:
            x_k = x_k + p_k
        print('x_%d' % (k_num+1))
        print(x_k)
        print('f_%d = %f'% (k_num+1, f(x_k)))
        print('gradient at x_%d = %f' %(k_num+1, np.linalg.norm(g_k, 2))) 
        k_num += 1

    print('After %d times iteration, the Minimum point is:'%(k_num))
    print(x_k)
    print('and the Minimum value is %f' %(f(x_k)))
    return x_k, f(x_k)


def PenaltySimple(f, c_eq, c_leq, *x_0, sigma = 0.01, alpha = 2, norm = 2, epsilon = 0.001):
    '''
    f is the target function, c_eq is a list contains equation constraints,
    c_leq is  a list contains unequal constrains, epsilon is the terminal parameter
    these functions could be function name or anonymous functions, which defined by 'lambda'
    '''
    #c_leq_new = []
    if not hasattr(f, "paraLength"):
        raise Exception("Please make sure that f has f.paraLength,  which is \
              the length of x_0,that is \"f.paraLength = len(x)\" ")
    assert isfunction(f)
    input_size  = f.paraLength
    # initializing x_0
    if len(x_0) != 0:
        x_0 = x_0[0]
    else:
        x_0 = np.zeros(input_size)
        
    x = tf.placeholder(tf.float32, shape=np.shape(x_0))
    sigma_tf = tf.placeholder(tf.float32)
    c_eq_new = []
    c_leq_new = []
    norm_tf = norm
    alpha_tf= alpha
    for fun in c_eq:
        c_eq_new.append(fun(x))
    for fun in c_leq:
        c_leq_new.append(tf.minimum(tf.constant(0.0), fun(x)))
    #c_leq_new = list(map(lambda fun:(lambda x:tf.minimum(0, fun(x))), c_leq))
    c_eq = c_eq_new + c_leq_new
    p_fun = f(x) + sigma_tf * tf.pow(tf.norm(c_eq, norm_tf), alpha_tf)
    c_xsigma_norm = 1000
    sess = tf.Session()
    k_num = 0
    while c_xsigma_norm > epsilon:
        ## steepest gradient methos
        print('The %dth iteration...'%(k_num+1))
        grad_inside = tf.gradients(p_fun, x)
        hess_inverse = tf.matrix_inverse(tf.hessians(p_fun,x))
        grad_norm = tf.norm(grad_inside, 2)
        grad_inside = sess.run(grad_inside, feed_dict = {x:x_0, sigma_tf:sigma})[0]
        hess_inverse = sess.run(hess_inverse , feed_dict = {x:x_0, sigma_tf:sigma})[0]
        grad_norm = sess.run(grad_norm, feed_dict = {x:x_0,sigma_tf:sigma})
        while grad_norm > 0.001:
            x_0 = x_0 - np.array((hess_inverse * np.mat(grad_inside).T).T.tolist())[0]
            grad_inside = tf.gradients(p_fun, x)
            hess_inverse = tf.matrix_inverse(tf.hessians(p_fun,x))
            grad_norm = tf.norm(grad_inside, 2)
            grad_inside = sess.run(grad_inside, feed_dict = {x:x_0, sigma_tf:sigma})[0]
            hess_inverse = sess.run(hess_inverse , feed_dict = {x:x_0, sigma_tf:sigma})[0]
            grad_norm = sess.run(grad_norm, feed_dict = {x:x_0,sigma_tf:sigma})
            print('grad_norm = %f'%(grad_norm))
        xsigma = copy.deepcopy(x_0)
        print('x_%d'%(k_num+1))
        print(x_0)
        c_xsigma_norm = sess.run(tf.norm(c_eq), feed_dict = {x:xsigma})
        sigma = 10*sigma
        k_num += 1
        print('gradient at x_%d =  %f'% (k_num, c_xsigma_norm))
    print('After %d iteration, the minimum point = '%(k_num))    
    print(xsigma)
    print('the minimum value = %f' %(f(xsigma)))
    print('c_i')
    print(sess.run(c_eq, feed_dict = {x:xsigma}))
    #p_fun = tf.norm(c_eq,2)
    #P_fun  = lambda x: (f(x) + sigma * 
    #         (np.linalg.norm(list(map(lambda fun:fun(x), c_eq)), norm) ** alpha))
    x_k = xsigma
    f_k = f(xsigma)
    return x_k, f_k
    
    
def main():    
    
    ## Demo 1:trust region method with subproblems solved by the Dogleg method
    print('Demo 1:trust region method with subproblems solved by the Dogleg method')
    f = lambda x:100*(x[1]-x[0]**2)**2 + (1-x[0])**2
    f.paraLength = 2   
#     print(f.paraLength)
    x_k, f_k = TrustRegion_dogleg(f, delta = 10)
    
    ## Demo 2:quasi-Newton method demo
    print('Demo 2:quasi-Newton method demo with BFGS')
    f = lambda x:x[0]**2 + 2 * x[1]**2
    f.paraLength = 2
    x_0 = np.array([1, 1])
    x_k, f_k = QuasiNewton(f, x_0, HUpdateMethod = 'BFGS')
    
    ## Demo 3:quasi-Newton method demo
    print('Demo 3:quasi-Newton method demo with DFP')
    f = lambda x:x[0]**2 + 2 * x[1]**2
    f.paraLength = 2
    x_0 = np.array([1, 1])
    x_k, f_k = QuasiNewton(f, x_0, HUpdateMethod = 'DFP')

    
    ## Demo 4:penalty function method demo
    print('Demo 4:penalty function method demo')
    f = lambda x:x[0] + x[1]
    f.paraLength = 2
    c_eq = [lambda x:x[0]**2 + x[1]**2 - 2]
    c_leq = []
    x_k, f_k = PenaltySimple(f, c_eq, c_leq, [-3,-4])
    #if len(c_leq) > 0:        
    
    ## Demo 5;CG method with exact line method
    print('Demo 5;CG method with exact line method')
    A = np.mat([[4, 1], [1,3]])
    print('A')
    print(A)
    x_0 = np.mat([2,1]).T
    print('x_0')
    print(x_0)
    b_vec = np.mat([-1, -2]).T
    print('b')
    print(b_vec)
    g_0 = A.T * x_0 + b_vec
    print('g_0')
    print(g_0)
    d_0 = -g_0
    print('d_0')
    print(d_0)
    beta_0 = 0
    k_num = 1
    while np.linalg.norm(g_0, 2) > 0.0001:
        print('The %dth iteration'%(k_num))
        alpha_0 = -(g_0.T * d_0)/(d_0.T * A * d_0)
        print('alpha_%d' %(k_num))
        print(alpha_0)
        x_1 = x_0 + d_0 * alpha_0
        print('x_%d'%(k_num))
        print(x_1)
        g_1 = A.T * x_1 + b_vec
        print('g_%d'%(k_num))
        print(g_1)
        beta_1 =  (g_1.T * A * d_0)/(d_0.T * A * d_0)
        print('beta_%d'%(k_num))
        print(beta_1)
        d_1 = -g_1 + d_0* beta_1 
        g_0 = g_1
        d_0 = d_1
        beta_0 = beta_1
        x_0 = x_1
        print('norm of gradients = %f'%(np.linalg.norm(g_0, 2)))
        print('f_%d = %f' %(k_num, 1/2 * x_0.T * A * x_0+b_vec.T*x_0))
        
if __name__ == "__main__":
    main()
    
    