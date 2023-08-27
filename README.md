------------------------------------------------------------------------------------------------------------------------------------------------------

This document gives a rough guide on how to use our code. If you have any questions, please email Robert at `rrs254@cornell.edu`. 
I split the code into a few .py files and a Jupyter notebook. 
For the most part, you should only need to interact with the Jupyter Notebook (`NDDE_1D.ipynb`). 
The repository currently has the following dependencies (In parenthesis, I list the version of each package that I currently use. The code does not need the specific versions of these libraries, but I included them just in case):

- `torch (1.13.1)`
- `matplotlib (3.5.1)`
- `seaborn (0.11.2)`
- `scipy (1.7.3)`
- `numpy (1.22.1)`
- `Jupyter`

------------------------------------------------------------------------------------------------------------------------------------------------------

All of our code has extensive comments. 
However, to give an overview, our library is structured as follows:

`MODEL.py`: This file houses class definitions for the Exponential and Logistic models. 
Both classes define a callable, parameterized torch.Module object that acts as the right-hand side of a DDE.

`Loss.py`: This file houses our loss functions. 
Currently, we have two loss functions: `SSE_Loss` and `Integral_Loss`. 
`SSE_Loss` is, as the name suggests, the sum of squares error between the target and predicted trajectory. 
The `Integral_Loss`, by contrast, implements a numerical approximation to the loss function $\int_{0}^{T} l(x(t)) dt$ (the equation just after equation (1) in the paper, but with $g = 0$). 
We use the trapezoidal rule to approximate the integral in this loss. 
All of our experiments in the paper use the `SSE_Loss`, but our algorithm works with both loss functions. 

`NDDE_1D.ipynb`: The main Jupyter Notebook file. 
This file drives the rest of the code. 
I'll talk about this file in more detail below. 

`NDDE.py`: This file houses three classes: `NDDE_1D`, `NDDE_Adjoint_SSE`, and `NDDE_Adjoint_l`. 
`NDDE_1D` is a wrapper around a Model object which calls one of the `NDDE_Adjoint` classes to make predictions and update the model's parameters. 
Both adjoint classes function similarly, so I will focus on `NDDE_Adjoint_SSE` (the one we used to get the results in the paper). 
This class is a `torch.autograd.Function` subclass which is designed to implement the forward and backward passes in our algorithm. 
The forward method takes a model, initial condition, $\tau$ estimate, $T$, and model parameters. 
It uses these values to compute the forward trajectory by solving the forward DDE using one of our solvers (see below). 
In other words, forward implements step 3 in our algorithm. 
It then stores the values we need for the backward step and returns the predicted trajectory. 

The backward method is more involved but essentially solves the adjoint equation and then uses this solution to compute the gradient of $L$ with respect to $\theta$ and $\tau$ (it returns these quantities). 
To do this, the backward method first fetches the data from the forward pass, sets up some tensors, and then solves the adjoint equation backward in time. 
We do this either using the forward Euler or RK2 solvers (The code is set up to the RK2 solver, though both solvers work). 
Once we have the discretized adjoint solution, we compute $dx(t_j)/d\theta$ and $dx(t_j)/d\tau$ for each $j$. 
We do this using the equations in the statement of theorem 1 of the paper. 
Finally, once we have these values, we can compute $dL/d\theta$ and $dL/d\tau$ using equation 4 in the paper. 
Many of these steps involve computing integrals. 
We use the trapezoidal rule to evaluate all integrals. 
This function produces several plots; they are for debugging purposes. 

`Solver.py`: This file houses two DDE solvers: 
A forward Euler solver and a basic Runge Kutta solver. 
We use these solvers for the forward pass in the DDE_Adjoint class. 
We used the Forward Euler solver to get the results in the paper, but both solvers work. 

------------------------------------------------------------------------------------------------------------------------------------------------------

With all that established, let's talk about how to use the Jupyter Notebook. 
The first two code cells import the relevant files/libraries. 
If you want to change the solver or loss function, you should do that here. 
Note that if you want to use a different solver, you will also need to change the corresponding import statement in the `NDDE.py` file. 
After importing, the cell titled "Generate Target Trajectory" uses the selected solver to get the target trajectory (by solving the true DDE, which is also set in this cell). 

Next, the cell titled "Setup, train the Model" initializes the model and runs the Epochs. 
This cell is where you change the initial guess for the parameters and $\tau$ (which - hopefully - will train to match the values you set for the true trajectory in the "Generate True Solution" cell). 
This cell also creates some buffers to track the past trajectories (for plotting purposes). 
For each epoch, we first compute the predicted trajectory using the current parameter, $\tau$ values. 
We then interpolate the target trajectory so we can evaluate the forward and adjoint trajectories at the same time values (in case the two use different step sizes). 
Next, we compute the loss between the predicted and target trajectories, perform backprop, and then update the Model's parameters and $\tau$. 
Note that the forward and backward passes use the NDDE_Adjoint class in `NDDE.py`. 

The final code cell of the notebook makes a few plots (including the loss history, target vs. predicted trajectories, and the loss as a function of $\tau$).

Using the setup in the code, our algorithm should converge in 144 epochs. 
Further, the final predicted values should be `tau = 0.99347, c_0 = 1.00325, c_1 = 1.00232`.

------------------------------------------------------------------------------------------------------------------------------------------------------

Hopefully, this is enough to get you started. 
If you have more specific questions about our implementation, our code's comments should be able to answer some of them. 
Otherwise, feel free to email me (`rrs254@cornell.edu`) with additional questions.

