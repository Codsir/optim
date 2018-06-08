## optim.py

### Based on: tensorflow, numpy, copy, inspect

### Why Tensorflow?

Tensorflow supports symbol computation well like Automatic derivation and the program
could be excuted with GPU, which will save our time.

#### dogleg(p_u, p_b, delta, tau = 2)

The Dogleg method to solve the subproblems of trust region method

#### getGrad(f, x_value)

Get the gradient of function f with *tf.gradients()*  <br />

```
    f= lambda x:100*(x[1]-x[0]**2)**2 + (1-x[0])**2
    x_value = [1.0,2.0]
    f_gradients = getGrad(f, x_value)
```

#### getHess(f, x_value)

Get the Hessian matrix of f with tf.hessian

#### TrustRegion_dogleg(f, delta = 0.5, eta = 0, *x_0, tolerance= 0.0001)

Trust region method with subproblems solved by the Dogleg method

#### ExactLineSearch_quadratic(f, x_k, p_k)

Exact line search method when the target function is quadratic

#### QuasiNewton(f, *x_0,  HUpdateMethod = 'BFGS', LineSearch = ExactLineSearch_quadratic, tolerance = 0.0001)

quasi-Newton method

#### PenaltySimple(f, c_eq, c_leq, epsilon)

f is the target function, c_eq is a list contains equation constraints,
c_leq is  a list contains unequal constrains, epsilon is the terminal parameter
these functions could be function name or anonymous functions, which defined by 'lambda'
The subproblem is solved by Newton Method, but it will be modified in the future because sometimes it's hard to compute        the inverse matrix of Hessian matrix.

### Example

#### Demo 1:trust region method with subproblems solved by the Dogleg method

```
    f = lambda x:100*(x[1]-x[0]**2)**2 + (1-x[0])**2
    f.paraLength = 2    ## 这一步不可缺少
    x_k, f_k = TrustRegion_dogleg(f, delta = 10)
```
    
#### Demo 2:quasi-Newton method demo

```
    print('Demo 2:quasi-Newton method demo')
    f = lambda x:x[0]**2 + 2 * x[1]**2
    f.paraLength = 2
    x_0 = np.array([1, 1])
    x_k, f_k = QuasiNewton(f, x_0)
```
    
#### Demo 3:penalty function method demo

```
    print('Demo 3:penalty function method demo')
    f = lambda x:x[0] + x[1]
    f.paraLength = 2
    c_eq = [lambda x:x[0]**2 + x[1]**2 - 2]
    c_leq = []
    x_k, f_k = PenaltySimple(f, c_eq, c_leq, [-3,-4])
```
