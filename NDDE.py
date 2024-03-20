import  torch; 
from    typing  import  Tuple, Callable;
from    scipy   import  interpolate;
import  matplotlib.pyplot as plt;

from    Solver  import  RK2             as DDE_Solver;

# Logger setup 
import logging;
LOGGER : logging.Logger = logging.getLogger(__name__);

# Should we make debug plots?
Debug_Plots : bool = False;



class NDDE_1D(torch.nn.Module):
    """
    Here, we define the NDDE_1D class. This class acts as a wrapper around a MODEL object. Recall 
    that a `MODEL` object acts like the function F in the following DDE:
            x'(t) = F(x(t), x(t - tau), t)          if t \in [0, T] 
            x(t)  = x_0                             if t \in [-tau, 0]
    The NDDE_1D class accepts a `MODEL`. Its forward method solves the implied DDE on the interval
    [0, T] and then returns the result.
    """
    
    def __init__(self, Model : torch.nn.Module):
        """
        Arguments:

        Model: This is a torch Module object which acts as the function "F" in a DDE: 
            x'(t) = F(x(t), x(t - \tau), t)     for t \in [0, T]
            x(t)  = x_0                         for t \in [-\tau, 0]
        Thus, the Model should accept three arguments: x(t), x(t - \tau), and t. 
        """

        # Call the super class initializer.
        super(NDDE_1D, self).__init__();

        # Set the model.
        self.Model = Model;
        

    def forward(self, x_0 : torch.Tensor, tau : torch.Tensor, T : torch.Tensor, l : Callable, G : Callable, x_Targ_Interp : Callable):
        """
        Arguments: 

        x_0: the initial position. We assume that x(t) = x_0 for t \in [-\tau, 0]. Thus, x_0 
        defines the initial state of the DDE. x_0 should be a 1D tensor.

        tau: The time delay in the DDE. This should be a single element tensor.

        T: The final time in our solution to the DDE (See above). This should be a single 
        element tensor.
    
        l: The function l in the loss function
            Loss(x_Pred) = \int_{0}^{T} l(x_Predict(t), x_Target(t)) dt
        Thus, it should be a Callable object which takes two arguments, both in R^d. We assume that
        this function can be differentiated (using autograd) with respect to its first argument.

        x_Targ_Interp: An interpolation object for the target trajectory. We need this to be able 
        to evaluate dl/dx when computing the adjoint in the backwards pass. 
        """

        # Run checks.
        assert(len(x_0.shape)   == 1);
        assert(tau.numel()      == 1);
        assert(T.numel()        == 1);

        # Fetch the model and its parameters.
        Model           : torch.nn.Module   = self.Model;
        Model_Params    : torch.Tensor      = Model.Params;

        # Evaluate the neural DDE using the Model
        #Trajectory = DDE_adjoint_Backward_SSE.apply(Model, x_0, tau, T, Model_Params);
        Trajectory = DDE_adjoint_Backward.apply(Model, x_0, tau, T, l, G, x_Targ_Interp, Model_Params);
        return Trajectory;



class DDE_adjoint_Backward(torch.autograd.Function):
    """
    This class implements the forward and backward passes for updating the parameters and tau. This
    particular class is designed for a loss function of the form
        \int_{0}^{T} l(x_Predict, x_Target) dx 

    Forward Pass - During the forward pass, we use a DDE solver to map the initial state, x_0, 
    along a predicted trajectory. In particular, we solve the following DDE
            x'(t)   = F(x(t), x(t - tau), t, theta) t \in [0, T]
            x(t)    = x_0                           t \in [-tau, 0]
    
    Backward pass - During the backward pass, we use the adjoint sensitivity method to find the 
    gradient of the loss with respect to tau and the network parameters. In particular, we solve
    the adjoint DDE and then use it to compute the gradients. 
    """

    @staticmethod
    def forward(ctx, 
                F               : torch.nn.Module, 
                x_0             : torch.Tensor, 
                tau             : torch.Tensor, 
                T               : torch.Tensor, 
                l               : Callable, 
                G               : Callable,
                x_Targ_Interp   : Callable, 
                F_Params        : torch.Tensor) -> torch.Tensor:
        """ 
        -------------------------------------------------------------------------------------------
        Arguments:

        F: A torch Module object which represents the right-hand side of the DDE,
                x'(t) = F(x(t), x(t - tau), t)      t \in [0, T]
        
        x_0: A d-element 1D tensor whose kth value specifies the kth component of the starting 
        position.

        tau: A single element tensor whose lone element specifies our best guess for the time 
        delay.

        T: A single element tensor whose lone element specifies the final simulation time.

        l: The function l in the loss function
            Loss(x_Pred) = G(x(T)) + \int_{0}^{T} l(x_Predict(t), x_Target(t)) dt
        Thus, it should be a Callable object which takes two arguments, both in R^d. We assume that
        this function can be differentiated (using autograd) with respect to its first argument.

        G: The function G in the loss function
            Loss(x_Pred) = G(x(T)) + \int_{0}^{T} l(x_Predict(t), x_Target(t)) dt
        Thus, it should be a Callable object which takes two arguments, both in R^d. We assume that
        this function can be differentiated (using autograd) with respect to its first argument.
        
        x_Target_Interp: An interpolation object for the target trajectory. We need this to be able 
        to evaluate dl/dx when computing the adjoint in the backwards pass. 

        F_Params: A tensor housing the parameters of the model, F.

        Note: we compute gradients with respect to x_0, tau, and F_Params.

        -------------------------------------------------------------------------------------------
        Returns: 

        A 2D tensor whose kth column specifies the state at the kth time step.        
        """ 

        # Run checks. 
        assert(len(x_0.shape)   == 1);
        assert(tau.numel()      == 1);
        assert(T.numel()        == 1);

        # We don't want gradients with respect to T.
        ctx.mark_non_differentiable(T);

        # Compute the forward solution using the DDE solver. 
        x_Trajectory, t_Trajectory = DDE_Solver(F, x_0, tau, T);
            
        # Save non-tensor arguments for backwards.
        ctx.F               = F; 
        ctx.x_Targ_Interp   = x_Targ_Interp;
        ctx.l               = l;
        ctx.G               = G;

        # Save tensor arguments for backwards
        ctx.save_for_backward(x_0, tau, T, x_Trajectory, t_Trajectory, F_Params);
        
        # All done!
        return x_Trajectory.detach();
        
    

    @staticmethod
    def backward(ctx, grad_y : torch.Tensor) -> Tuple[torch.Tensor]:
        # recover information from the forward pass
        F               : torch.nn.Module                   = ctx.F;
        x_Targ_Interp   : Callable                          = ctx.x_Targ_Interp;
        l               : Callable                          = ctx.l;
        G               : Callable                          = ctx.G;
        x_0, tau, T, x_Trajectory, t_Trajectory, F_Params   = ctx.saved_tensors;
        d               : int                               = x_0.numel();

        # extract the parameters from the list and find how many there are
        F_Params        : torch.Tensor      = F.Params;
        N_Params        : int               = F_Params.shape[0];
        
        # Find the step size for the backwards pass. Also find the number of time step and the 
        # number of time steps in an interval of length tau. 
        N_tau           : int               = 10;
        dt              : float             = tau.item()/N_tau;
        N               : int               = int(torch.floor(T/dt).item());

        # Now, let's set up an interpolation of the forward trajectory. We will evaluate this 
        # interpolation at each time when we want to compute the adjoint. This allows us to use
        # a different time step for the forward and backwards passes. 
        x_Pred_Interp                       = interpolate.interp1d(x = t_Trajectory.detach().numpy(), y = x_Trajectory.detach().numpy())

        # Find time values for backwards pass. 
        t_Values        : torch.Tensor      = torch.linspace(start = 0, end = T, steps = N + 1);

        # evaluate the interpolation of the predicted, target solution at these values. 
        x_Targ_Values   : torch.Tensor      = torch.from_numpy(x_Targ_Interp(t_Values.detach().numpy()));
        x_Pred_Values   : torch.Tensor      = torch.from_numpy(x_Pred_Interp(t_Values.detach().numpy()));

        # Set up a tensor to hold the adjoint. p[:, j] holds the value of the adjoint at the
        # jth time value.   
        p               : torch.Tensor      = torch.zeros([d, N + 1],    dtype = torch.float32);

        # Now, we need to set p's initial conditions. From the paper, the adjoint at time T should 
        # be set to -dg/dx(T). Let's compute that! Note that it's possible that our implementation 
        # of G doesn't directly depend on x(T) (it may just return a zero vector). In this case, 
        # \nabla_{x(T)} G(x(T)) will return None. If we get None, we just set the gradient to zero.
        torch.set_grad_enabled(True);

        xT_Predict      : torch.Tensor  = x_Pred_Values[:, -1].requires_grad_(True);
        xT_Target       : torch.Tensor  = x_Targ_Values[:, -1];
        G_xT            : torch.Tensor  = G(xT_Predict, xT_Target);

        grad_G_xT   : torch.Tensor  = torch.autograd.grad(outputs = G_xT, inputs = xT_Predict, allow_unused = True)[0];
        if(grad_G_xT is None):
            grad_G_xT = torch.zeros_like(xT_Predict);
        p[:, -1] = -grad_G_xT;
        
        torch.set_grad_enabled(False);

        # Set up vectors to track (dF_dx(t))^T p(t), (dF_dy(t))^T p(t), (dF_dTheta(t))^T p(t), 
        # and (dF_dy(t + tau))F(t).                                                 # do we need to compute 1 time step ahead?
        F_Values        = torch.zeros([d,           N + 1], dtype = torch.float32); # no
        p_t_dFdx_t      = torch.zeros([d,           N + 1], dtype = torch.float32); # no
        p_t_dFdy_t      = torch.zeros([d,           N + 1], dtype = torch.float32); # yes
        p_t_dFdTheta_t  = torch.zeros([N_Params,    N + 1], dtype = torch.float32); # no
        dldx_t          = torch.zeros([d,           N + 1], dtype = torch.float32); # yes



        ###########################################################################################
        # Compute vector, jacobian products at each time step.

        # Since we use the RK2 method, we actually need to compute p(t + tau) (dF_dy)(t + tau) and 
        # (dl/dx)(t) one time step ahead. Crucially, since we don't use the former for the first 
        # few time steps, we can safely ignore it. We do need to compute the former, however.
        torch.set_grad_enabled(True);
        
        xT_Predict      : torch.Tensor  = x_Pred_Values[:, -1].requires_grad_(True);
        xT_Target       : torch.Tensor  = x_Targ_Values[:, -1];
        l_xT            : torch.Tensor  = l(xT_Predict, xT_Target);
        dldx_t[:, -1]                   = torch.autograd.grad(outputs = l_xT, inputs = xT_Predict)[0];

        torch.set_grad_enabled(False);

        # Solve the adjoint equation backwards in time.
        for j in range(N, 0, -1):  
            # Enable gradient tracking! We need this to compute the gradients of F. 
            torch.set_grad_enabled(True);

            # First, let's compute p(t) dF_dx(t), p(t) dF_dy(t), and p(t) dF_dtheta(t).
            t_j         : torch.Tensor  = t_Values[j];
            x_j         : torch.Tensor  = x_Pred_Values[:, j].requires_grad_(True);
            y_j         : torch.Tensor  = x_Pred_Values[:, j - N_tau].requires_grad_(True) if j - N_tau >= 0 else x_0;
            p_j         : torch.Tensor  = p[:, j];

            F_j         : torch.Tensor  = F(x_j, y_j, t_j);
            F_Values[:, j]              = F_j;

            p_t_dFdx_t[:, j], p_t_dFdy_t[:, j], p_t_dFdTheta_t[:, j] = torch.autograd.grad(
                                                                            outputs         = F_j, 
                                                                            inputs          = (x_j, y_j, F_Params), 
                                                                            grad_outputs    = p_j);
            
            # Compute dl_dx at the j-1'th time step.
            x_Targ_jm1  : torch.Tensor  = x_Targ_Values[:, j - 1];
            x_jm1       : torch.Tensor  = x_Pred_Values[:, j - 1].requires_grad_(True);
            l_x_jm1     : torch.Tensor  = l(x_jm1, x_Targ_jm1);
            dldx_t[:, j - 1]             = torch.autograd.grad(outputs = l_x_jm1, inputs = x_jm1)[0];
    
            # We are all done tracking gradients.
            torch.set_grad_enabled(False);
        
            # Find p at the previous time step. Recall that p satisfies the following DDE:
            #       p'(t) = -dF_dx(t)^T p(t)  - dF_dy(t)^T p(t + tau) 1_{t + tau < T}(t) + d l(x(t))/d x(t)
            #       p(T)  = dG_dX(x(T))  
            # Let F(t, p(t), p(t + tau)) denote the right-hand side of this equation. We find a 
            # numerical solution to this DDE using the RK2 method. In this case, we compute
            #       p(t - dt) \approx p(t) - dt*0.5*(k1 + k2)
            #       k1 = -dF_dx(t)^T p(t)  - dF_dy(t)^T p(t + tau) 1_{t + tau < T}(t) + d l(x(t))/d x(t)
            #       k2 = -dF_dx(t - dt)^T [p(t) - dt*k1]  - dF_dy(t - dt)^T p(t - dt + tau) 1_{t - dt + tau < T}(t) + d l(x(t - dt))/d x(t - dt)
            """
            k1 : torch.Tensor   = -torch.mv(torch.transpose(dF_dx[:, :, j], 0, 1), p[:, j]) + dl_dx[:, j];
            if(j + N_tau < N):
                k1 += -torch.mv(torch.transpose(dF_dy[:, :, j + N_tau], 0, 1), p[:, j + N_tau]);
            
            k2 : torch.Tensor   = -torch.mv(torch.transpose(dF_dx[:, :, j - 1], 0, 1), (p[:, j] - dt*k1)) + dl_dx[:, j - 1];
            if(j - 1 + N_tau < N):
                k2 += -torch.mv(torch.transpose(dF_dy[:, :, j - 1 + N_tau], 0, 1), p[:, j - 1 + N_tau]);

            p[:, j - 1] = p[:, j] - dt*0.5*(k1 + k2);
            """
            if(j + N_tau >= N):
                p[:, j - 1] = p[:, j] - dt*(-p_t_dFdx_t[:, j] + dldx_t[:, j]);
            else: 
                p[:, j - 1] = p[:, j] - dt*(-p_t_dFdx_t[:, j] + dldx_t[:, j] - p_t_dFdy_t[:, j + N_tau]);



        ###########################################################################################
        # Compute dL_dtau and dL_dtheta.
        
        # Set up vectors to hold dL_dtau and dL_dtheta
        dL_dtheta       : torch.Tensor      = torch.zeros_like(F_Params);
        dL_dtau         : torch.Tensor      = torch.tensor([0.], dtype = torch.float32);
        dL_dx0          : torch.Tensor      = torch.zeros_like(x_0);

        # Now compute dL_dtheta and dL_dtau. In this case, 
        #   dL_dtheta   =  -\int_{t = 0}^T dF_dtheta(x(t), x(t - tau), t)^T p(t) dt
        #   dL_dtau     = \int_{t = 0}^{T - tau} dF_dy(x(t + tau), x(t), t)^T p(t + tau) \cdot F(x(t), x(t - tau), t) dt
        #   dL_dx0      = -p[0] -\int_{t = 0}^{-tau} dF_dy(x(t), x(t - \tau), t)^T p(t) dt
        # We compute these integrals using the trapezoidal rule.
        for j in range(0, N):
            dL_dtheta   -=  0.5*dt*(p_t_dFdTheta_t[:, j] + p_t_dFdTheta_t[:, j + 1]);
        for j in range(0, N - N_tau):
            dL_dtau     +=  0.5*dt*(torch.dot(F_Values[:, j    ], p_t_dFdy_t[:, j + N_tau]) + 
                                    torch.dot(F_Values[:, j + 1], p_t_dFdy_t[:, j + 1 + N_tau]));
        for j in range(0, N_tau):
            dL_dx0      -=  0.5*dt*(p_t_dFdy_t[:, j] + p_t_dFdy_t[:, j + 1]);
        
        dL_dx0 -= p[:, 0];

        # All done... The kth return argument represents the gradient for the kth argument to forward.
        return None, dL_dx0, dL_dtau, None, None, None, None, dL_dtheta;
