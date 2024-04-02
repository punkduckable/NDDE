import  torch; 
from    typing  import  Tuple, Callable, List;
from    scipy   import  interpolate;
import  matplotlib.pyplot as plt;

from    Solver  import  RK2             as DDE_Solver;

# Logger setup 
import logging;
LOGGER : logging.Logger = logging.getLogger(__name__);

# Should we make debug plots?
Debug_Plots : bool = False;



class NDDE(torch.nn.Module):
    """
    Here, we define the NDDE class. This class acts as a wrapper around a MODEL object. Recall 
    that a `MODEL` object acts like the function F in the following DDE:
            x'(t) = F(x(t), x(t - tau), tau, t)         if t \in [0, T] 
            x(t)  = X0(t)                               if t \in [-tau, 0]
    The NDDE class accepts a `MODEL`. Its forward method solves the implied DDE on the interval
    [0, T] and then returns the result.
    """
    
    def __init__(self, F : torch.nn.Module, X0 : torch.nn.Module) -> None:
        """
        -------------------------------------------------------------------------------------------
        Arguments:
        -------------------------------------------------------------------------------------------

        F: This is a torch Module object which acts as the function "F" in a DDE: 
            x'(t) = F(x(t), x(t - \tau), tau, t)    for t \in [0, T]
            x(t)  = X0(t)                           for t \in [-\tau, 0]
        Thus, the F should accept three arguments: x(t), x(t - \tau), and t. 
        
        X0: This is a module which gives the initial condition in [-\tau, 0]. In particular, at
        time t \in [-\tau, 0], we set the initial condition at time t to X0(t). X0 should be a
        torch.nn.Module object which takes a tensor (of shape S) of time values and returns a S x d
        tensor[s, :] element holds the value of the IC at the s'th element of the input tensor.
        """

        # Call the super class initializer.
        super(NDDE, self).__init__();

        # Store the right hand side and the IC.
        self.F      = F;
        self.X0     = X0;
        


    def forward(self, tau : torch.Tensor, T : torch.Tensor, l : torch.nn.Module, G : torch.nn.Module, x_Targ_Interp : Callable):
        """
        -------------------------------------------------------------------------------------------
        Arguments: 
        -------------------------------------------------------------------------------------------
        
        tau: The time delay in the DDE. This should be a single element tensor.

        T: The final time in our solution to the DDE (See above). This should be a single 
        element tensor.
    
        l: The function l in the loss function
            Loss(x_Pred) = G(x(T)) + \int_{0}^{T} l(x_Predict(t), x_Target(t)) dt
        Thus, it should be a torch.nn.Module  object which takes two arguments, both in R^d. We 
        assume that this function can be differentiated (using autograd) with respect to its first 
        argument.

        G: The function G in the loss function
            Loss(x_Pred) = G(x(T)) + \int_{0}^{T} l(x_Predict(t), x_Target(t)) dt
        Thus, it should be a torch.nn.Module object which takes two arguments, both in R^d. We 
        assume that this function can be differentiated (using autograd) with respect to its first 
        argument.

        x_Targ_Interp: An interpolation object for the target trajectory. We need this to be able 
        to evaluate dl/dx when computing the adjoint in the backwards pass. 
        """

        # Run checks.
        assert(tau.numel()      == 1);
        assert(T.numel()        == 1);

        # Fetch F, X0, and their parameters.
        F           : torch.nn.Module       = self.F;
        X0          : torch.nn.Module       = self.X0;

        Params      : List[torch.Tensor]    = [];
        N_F_Params  : int                   = 0;
        for Param in F.parameters():
            Params.append(Param);
            N_F_Params += 1;
        
        N_X0_Params : int                   = 0;
        for Param in X0.parameters():
            Params.append(Param);
            N_X0_Params += 1;

        # Evaluate the neural DDE using F, tau, and X0.
        Trajectory : torch.Tensor = DDE_adjoint_Backward.apply(F, X0, tau, T, l, G, x_Targ_Interp, N_F_Params, N_X0_Params, *Params);
        return Trajectory;



class DDE_adjoint_Backward(torch.autograd.Function):
    """
    This class implements the forward and backward passes for updating the parameters and tau. This
    particular class is designed for a loss function of the form
        \int_{0}^{T} l(x_Predict, x_Target) dx 

    Forward Pass - During the forward pass, we use a DDE solver to map the initial state, X0, 
    along a predicted trajectory. In particular, we solve the following DDE
            x'(t)   = F(x(t), x(t - tau), tau, t)       t \in [0, T]
            x(t)    = X0(t)                             t \in [-tau, 0]
    
    Backward pass - During the backward pass, we use the adjoint sensitivity method to find the 
    gradient of the loss with respect to tau and the network parameters. In particular, we solve
    the adjoint DDE and then use it to compute the gradients. 
    """

    @staticmethod
    def forward(ctx, 
                F               : torch.nn.Module, 
                X0              : torch.Tensor, 
                tau             : torch.Tensor, 
                T               : torch.Tensor, 
                l               : torch.nn.Module, 
                G               : torch.nn.Module,
                x_Targ_Interp   : Callable, 
                N_F_Params      : int,
                N_X0_Params     : int,
                *Params         : torch.Tensor) -> torch.Tensor:
        """ 
        -------------------------------------------------------------------------------------------
        Arguments:

        F: A torch Module object which represents the right-hand side of the DDE,
                x'(t) = F(x(t), x(t - tau), tau, t)      t \in [0, T]
        
        X0: This is a module which gives the initial condition in [-\tau, 0]. In particular, at
        time t \in [-\tau, 0], we set the initial condition at time t to X0(t). X0 should be a
        torch.nn.Module object which takes a tensor (of shape S) of time values and returns a S x d
        tensor[s, :] element holds the value of the IC at the s'th element of the input tensor.

        tau: A single element tensor whose lone element specifies our best guess for the time 
        delay.

        T: A single element tensor whose lone element specifies the final simulation time.

        l: The function l in the loss function
            Loss(x_Pred) = G(x(T)) + \int_{0}^{T} l(x_Predict(t), x_Target(t)) dt
        Thus, it should be a torch.nn.Module  object which takes two arguments, both in R^d. We 
        assume that this function can be differentiated (using autograd) with respect to its first 
        argument.

        G: The function G in the loss function
            Loss(x_Pred) = G(x(T)) + \int_{0}^{T} l(x_Predict(t), x_Target(t)) dt
        Thus, it should be a torch.nn.Module object which takes two arguments, both in R^d. We 
        assume that this function can be differentiated (using autograd) with respect to its first 
        argument.
        
        x_Target_Interp: An interpolation object for the target trajectory. We need this to be able 
        to evaluate dl/dx when computing the adjoint in the backwards pass. 

        N_F_Params: The number of tensor parameters in F. The first N_F_Params elements of Params 
        should hold F's parameters (each one of which is a tensor).

        N_X0_Params: The number of tensor parameters in X0. The last N_X0_Params elements of 
        Params should X0's parameters (each one of which is a tensor).

        Params: A list housing all of the parameters in F and X0. The first N_F_Params elements of 
        this list should hold F's parameters, while the remaining N_X0_Params should hold X0's 
        parameters.

                
        -------------------------------------------------------------------------------------------
        Returns: 
        
        We compute and return gradients with respect to tau and Params (both F's and X0's).       
        """ 

        # Run checks. 
        assert(tau.numel()      == 1);
        assert(T.numel()        == 1);
        assert(len(Params)      == N_F_Params + N_X0_Params);

        # Issue a warning if T < tau. In this case, we can still technically compute everything,
        # but there will be no dependence on tau.
        if(tau.item() > T.item()):
            LOGGER.warning("T = %f < %f = tau. The forward trajectory (and, therefore, the loss) does not depend on tau" % (T.item(), tau.item()));

        # We don't want gradients with respect to T.
        ctx.mark_non_differentiable(T);

        # Compute the forward solution using the DDE solver. 
        x_Trajectory, t_Trajectory = DDE_Solver(F, X0, tau, T);
            
        # Save non-tensor arguments for backwards.
        ctx.F               = F; 
        ctx.X0              = X0;
        ctx.x_Targ_Interp   = x_Targ_Interp;
        ctx.l               = l;
        ctx.G               = G;
        ctx.N_F_Params      = N_F_Params;
        ctx.N_X0_Params     = N_X0_Params;

        # Save tensor arguments for backwards
        ctx.save_for_backward(tau, T, x_Trajectory, t_Trajectory, *Params);
        
        # All done!
        return x_Trajectory.detach();
        
    

    @staticmethod
    def backward(ctx, grad_y : torch.Tensor) -> Tuple[torch.Tensor]:
        ###########################################################################################
        # Setup

        # recover information from the forward pass
        F               : torch.nn.Module                   = ctx.F;
        X0              : torch.nn.Module                   = ctx.X0;
        x_Targ_Interp   : Callable                          = ctx.x_Targ_Interp;
        l               : torch.nn.Module                   = ctx.l;
        G               : torch.nn.Module                   = ctx.G;
        N_F_Params      : int                               = ctx.N_F_Params;
        N_X0_Params     : int                               = ctx.N_X0_Params; 
        tau, T, x_Trajectory, t_Trajectory, *Params         = ctx.saved_tensors;

        # Fetch F's and X0's parameters
        F_Params        : int               = Params[:N_F_Params];
        X0_Params       : int               = Params[N_F_Params:];

        # Find the step size for the backwards pass. Also find the number of time step and the 
        # number of time steps in an interval of length tau. 
        N_tau           : int               = 10;
        dt              : float             = tau.item()/N_tau;
        N               : int               = int(torch.round(T/dt).item());

        # Now, let's set up an interpolation of the forward trajectory. We will evaluate this 
        # interpolation at each time when we want to compute the adjoint. This allows us to use
        # a different time step for the forward and backwards passes. 
        x_Pred_Interp                       = interpolate.interp1d(x = t_Trajectory.detach().numpy(), y = x_Trajectory.detach().numpy())
        
        # Find time values for backwards pass. 
        t_Values        : torch.Tensor      = torch.linspace(start = 0, end = T, steps = N + 1);

        # evaluate the interpolation of the predicted, target solution at these values. 
        x_Targ_Values   : torch.Tensor      = torch.from_numpy(x_Targ_Interp(t_Values.detach().numpy()));
        x_Pred_Values   : torch.Tensor      = torch.from_numpy(x_Pred_Interp(t_Values.detach().numpy()));
        
        # Determine the dimension of the space in which the dynamics happen
        d               : int               = x_Pred_Values.shape[0];

        # Determine the time values at which we will evaluate the IC: -\tau, -\tau + dt, ... , 0.
        t_Values_X0     : torch.Tensor      = torch.linspace(start = -tau.item(), end = 0, steps = N_tau + 1);



        ###########################################################################################
        # Initialize arrays to hold p, vector-jacobian products at each time step

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

        grad_G_xT   : torch.Tensor  = torch.autograd.grad(  outputs         = G_xT, 
                                                            inputs          = xT_Predict, 
                                                            allow_unused    = True)[0];
        if(grad_G_xT is None):
            grad_G_xT = torch.zeros_like(xT_Predict);
        p[:, -1] = -grad_G_xT;
        
        torch.set_grad_enabled(False);

        # Set up vectors to track (dF_dx(t))^T p(t), (dF_dy(t))^T p(t), (dF_dy(t + tau))F(t), 
        # (dF_dTheta)(t))^T p(T), (dF_dtau)(t)^T p(t) and (dX0_dPhi(t - tau))^T (dF_dY(t)^T p(t)).
        F_Values            = torch.empty([d,           N + 1], dtype = torch.float32);
        dFdx_T_p            = torch.empty([d,           N + 1], dtype = torch.float32);
        dFdy_T_p            = torch.empty([d,           N + 1], dtype = torch.float32);
        dFdtau_T_p          = torch.empty([d,           N + 1], dtype = torch.float32);
        dldx                = torch.empty([d,           N + 1], dtype = torch.float32);
        
        dFdTheta_T_p_tj      = [];
        for i in range(N_F_Params):
            dFdTheta_T_p_tj.append(torch.empty(list(F_Params[i].shape) + [N + 1], dtype = torch.float32));

        dX0dPhi_T_dFdy_T_p  = [];
        for i in range(N_X0_Params):
            dX0dPhi_T_dFdy_T_p.append(torch.empty(list(X0_Params[i].shape) + [N_tau + 1], dtype = torch.float32));
        
        # Set up vectors to hold dL_dtau, dL_dTheta, and dL_dPhi.
        dL_dTheta       : List[torch.Tensor] = [];
        for i in range(N_F_Params):
            dL_dTheta.append(torch.zeros_like(F_Params[i]));
        dL_dtau         : torch.Tensor      = torch.zeros(1, dtype = torch.float32);
        dL_dPhi         : List[torch.Tensor] = [];
        for i in range(N_X0_Params):
            dL_dPhi.append(torch.zeros_like(X0_Params[i]));

        # Initialize the adjoint, theta derivative calculations from the j+1'th time step (when 
        # j == N, this is just zero).
        dFdTheta_T_p_tjp1    : List[torch.Tensor] = [];
        for i in range(N_F_Params):
            dFdTheta_T_p_tjp1.append(torch.empty_like(F_Params[i]));


        ###########################################################################################
        # Compute vector, jacobian products at each time step.

        # Since we use the RK2 method, we actually need to compute p(t + tau) (dF_dy)(t + tau) and 
        # (dl/dx)(t) one time step ahead. Crucially, since we don't use the former for the first 
        # few time steps, we can safely ignore it. We do need to compute the former, however.
        torch.set_grad_enabled(True);
        
        xT_Predict      : torch.Tensor  = x_Pred_Values[:, -1].requires_grad_(True);
        xT_Target       : torch.Tensor  = x_Targ_Values[:, -1];
        l_xT            : torch.Tensor  = l(xT_Predict, xT_Target);
        dldx[:, -1]                     = torch.autograd.grad(outputs = l_xT, inputs = xT_Predict)[0];

        torch.set_grad_enabled(False);


        # Solve the adjoint equation backwards in time.
        for j in range(N, 0, -1):  
            # -------------------------------------------------------------------------------------
            # Compute Vector-Jacobian products. 

            # Enable gradient tracking! We need this to compute the gradients of F. 
            torch.set_grad_enabled(True);

            # First, let's compute p(t) dF_dx(t), p(t) dF_dy(t), and p(t) dF_dtheta(t).
            t_j         : torch.Tensor  = t_Values[j];
            x_j         : torch.Tensor  = x_Pred_Values[:, j].requires_grad_(True);
            y_j         : torch.Tensor  = x_Pred_Values[:, j - N_tau].requires_grad_(True) if j - N_tau >= 0 else X0(t_j - tau).detach().reshape(-1).requires_grad_(True);
            p_j         : torch.Tensor  = p[:, j];

            F_j         : torch.Tensor  = F(x_j, y_j, tau, t_j);
            F_Values[:, j]              = F_j;

            dFdx_T_p[:, j], dFdy_T_p[:, j], dFdtau_T_p_tj, *dFdTheta_T_p_tj = torch.autograd.grad(
                                                                            outputs         = F_j, 
                                                                            inputs          = (x_j, y_j, tau, *F_Params), 
                                                                            grad_outputs    = p_j, 
                                                                            allow_unused    = True);
            
            # F may not explicitly depend on tau, in which case dFdtau_T_p_tj will be None.
            if(dFdtau_T_p_tj is None):
                dFdtau_T_p[:, j] = torch.zeros_like(F_j);
            else:
                dFdtau_T_p[:, j] = dFdtau_T_p_tj

            # Compute dl_dx at the j-1'th time step.
            x_Targ_jm1  : torch.Tensor  = x_Targ_Values[:, j - 1];
            x_jm1       : torch.Tensor  = x_Pred_Values[:, j - 1].requires_grad_(True);
            l_x_jm1     : torch.Tensor  = l(x_jm1, x_Targ_jm1);
            dldx[:, j - 1]              = torch.autograd.grad(outputs = l_x_jm1, inputs = x_jm1)[0];
    
            # Compute (dX0_dPhi(t - tau))^T (dF_dy(t)^T p(t)).
            if(j <= N_tau):
                # Evaluate X0 at t_j.
                tj_m_tau        : torch.Tensor = t_Values_X0[j];
                X0_tj_m_tau     : torch.Tensor = X0(tj_m_tau).reshape(-1);

                # Compute the gradient of the with respect to it's parameters at time tj.
                dX0dPhi_T_dFdY_T_p_tj          = torch.autograd.grad(
                                                        outputs         = X0_tj_m_tau,
                                                        inputs          = X0_Params,
                                                        grad_outputs    = dFdy_T_p[:, j]);
                
                # Store the gradients!
                for i in range(N_X0_Params):
                    dX0dPhi_T_dFdy_T_p[i][..., j] = dX0dPhi_T_dFdY_T_p_tj[i];

            # We are all done tracking gradients.
            torch.set_grad_enabled(False);
            
            
            # -------------------------------------------------------------------------------------
            # Update p

            # Find p at the previous time step. Recall that p satisfies the following DDE:
            #       p'(t) = -dF_dx(t)^T p(t)  - dF_dy(t + tau)^T p(t + tau) 1_{t + tau < T}(t) + d l(x(t))/d x(t)
            #       p(T)  = dG_dX(x(T))  
            # We find a numerical solution to this DDE using the RK2 method. In this case, we 
            # compute
            #       p(t - dt) \approx p(t) - dt*0.5*(k1 + k2)
            #       k1 = -dF_dx(t)^T p(t)  - dF_dy(t + tau)^T p(t + tau) 1_{t + tau < T}(t) + d l(x(t))/d x(t)
            #       k2 = -dF_dx(t - dt)^T [p(t) - dt*k1]  - dF_dy(t + tau - dt)^T p(t - dt + tau) 1_{t - dt + tau < T}(t) + d l(x(t - dt))/d x(t - dt)
            
            # First, compute k1
            k1 : torch.Tensor   = -dFdx_T_p[:, j] + dldx[:, j]
            if(j + N_tau < N):
                k1 -= dFdy_T_p[:, j + N_tau];

            # To compute k2, we need to first compute the Jacobian Vector product -dF_dx(t - dt)^T [p(t) - dt*k1].
            torch.set_grad_enabled(True);

            t_jm1               : torch.Tensor  = t_Values[j - 1];
            x_jm1               : torch.Tensor  = x_Pred_Values[:, j - 1].requires_grad_(True);
            y_jm1               : torch.Tensor  = x_Pred_Values[:, j - 1 - N_tau].requires_grad_(False) if j - 1 - N_tau >= 0 else X0(t_jm1 - tau).detach().reshape(-1).requires_grad_(False);
            F_jm1               : torch.Tensor  = F(x_jm1, y_jm1, tau, t_jm1);
            p_k1_t_dFdx_t       : torch.Tensor  = torch.autograd.grad(  outputs         = F_jm1, 
                                                                        inputs          = x_jm1, 
                                                                        grad_outputs    = p_j - dt*k1)[0];

            torch.set_grad_enabled(False);

            # We can now compute k2
            k2 : torch.Tensor   = -p_k1_t_dFdx_t + dldx[:, j - 1];
            if(j - 1 + N_tau < N):
                k2 -= dFdy_T_p[:, j - 1 + N_tau];

            # Finally, we can compute p.
            p[:, j - 1] = p[:, j] - dt*0.5*(k1 + k2);
            
            """
            # Forward Euler Method
            if(j + N_tau >= N):
                p[:, j - 1] = p[:, j] - dt*(-dFdx_T_p[:, j] + dldx[:, j]);
            else: 
                p[:, j - 1] = p[:, j] - dt*(-dFdx_T_p[:, j] + dldx[:, j] - dFdy_T_p[:, j + N_tau]);
            """

            # -------------------------------------------------------------------------------------
            # Accumulate results to compute integrals for dL_dtau, dL_dTheta, and dL_dPhi
    
            # In this case, 
            #   dL_dTheta   =  -\int_{t = 0}^T dF_dTheta(x(t), x(t - tau), tau, t)^T p(t) dt
            #
            #   dL_dtau     = \int_{t = 0}^{T - tau} dF_dy(x(t + tau), x(t), tau, t + tau)^T p(t + tau) \cdot F(x(t), x(t - tau), tau, t) dt 
            #               + \int_{0}^{T} p(t) \cdot (dF_dtau)(x(t), x(t - tau), tau, t) dt
            #
            #   dL_dPhi     = -(dX0_dPhi(0))^T p(0) -\int_{t = 0}^{tau} (dX0_dPhi(t - tau))^T (dF_dy(x(t), x(t - tau), tau, t)^T p(t)) dt
            # We compute these integrals using the trapezoidal rule. Here, we add the contribution
            # from the j'th step.

            # dL_dtau
            if(j < N):
                dL_dtau     -=  0.5*dt*(dFdtau_T_p[:, j] + dFdtau_T_p[:, j + 1]);
            if(j < N - N_tau):
                dL_dtau     +=  0.5*dt*(torch.dot(dFdy_T_p[:, j +     N_tau], F_Values[:, j    ]) + 
                                        torch.dot(dFdy_T_p[:, j + 1 + N_tau], F_Values[:, j + 1]));
            # dL_dTheta 
            if(j < N):
                for i in range(N_F_Params):
                    dL_dTheta[i] -= 0.5*dt*(dFdTheta_T_p_tj[i] + dFdTheta_T_p_tjp1[i]);
            
            # dL_dPhi
            if(j < N_tau):
                for i in range(N_X0_Params):
                    dL_dPhi[i]  -= 0.5*dt*(dX0dPhi_T_dFdy_T_p[i][..., j] + dX0dPhi_T_dFdy_T_p[i][..., j + 1]);


            # -------------------------------------------------------------------------------------
            # Update dFdTheta_T_p_tjp1.
        
            for i in range(N_F_Params):
                dFdTheta_T_p_tjp1[i] = dFdTheta_T_p_tj[i];



        ###########################################################################################
        # Compute jacobian, vector products at t = 0.

        # Enable gradient tracking! We need this to compute the gradients of F. 
        torch.set_grad_enabled(True);

        # First, let's compute p(0) dF_dx(0), p(0) dF_dy(0), and p(0) dF_dTheta(0).
        t_0         : torch.Tensor  = t_Values[0];
        x_0         : torch.Tensor  = x_Pred_Values[:, 0].requires_grad_(True);
        y_0         : torch.Tensor  = X0(t_Values_X0[0]).reshape(-1).requires_grad_(True);
        p_0         : torch.Tensor  = p[:, 0];

        F_0         : torch.Tensor  = F(x_0, y_0, tau, t_0);
        F_Values[:, 0]              = F_0;

        dFdx_T_p[:, 0], dFdy_T_p[:, 0], dFdtau_T_p_t0, *dFdTheta_T_p_t0 = torch.autograd.grad(
                                                                        outputs         = F_0, 
                                                                        inputs          = (x_0, y_0, tau, *F_Params), 
                                                                        grad_outputs    = p_0, 
                                                                        allow_unused    = True);

        # F may not explicitly depend on tau, in which case dFdtau_T_p_t0 will be None.
        if(dFdtau_T_p_t0 is None):
            dFdtau_T_p[:, 0] = torch.zeros_like(F_0);
        else:
            dFdtau_T_p[:, 0] = dFdtau_T_p_t0


        # Now, let's compute (dX0_dPhi(-tau))^T (dF_dy(0)^T p(0)).
        dX0dPhi_T_dFdy_T_p_t0 = torch.autograd.grad(                    outputs         = X0(t_Values_X0[0]).reshape(-1),
                                                                        inputs          = X0_Params, 
                                                                        grad_outputs    = dFdy_T_p[:, 0]);

        # Finally, let's compute (dX0_dPhi(0))^T p(0)
        dX0dPhi_T_p_t0       = torch.autograd.grad(                     outputs         = X0(t_0).reshape(-1), 
                                                                        inputs          = X0_Params,
                                                                        grad_outputs    = p_0);

        # And finally, let's store the gradients. 
        for i in range(N_X0_Params):
            dX0dPhi_T_dFdy_T_p[i][..., 0] = dX0dPhi_T_dFdy_T_p_t0[i];

        torch.set_grad_enabled(False);
        

        # -----------------------------------------------------------------------------------------
        # Accumulate the contributions to the gradient integral from the 0 time step. 

        # dL_dtau 
        dL_dtau -= 0.5*dt*(dFdtau_T_p[:, 0] + dFdtau_T_p[:, 1]);

        # Note: If tau > T, then the loss has no dependence on tau. Otherwise, we should accumulate 
        # the portion of the integral for dL_dtau from the interval [0, dt].
        if(N_tau < N):
            dL_dtau     +=  0.5*dt*(torch.dot(dFdy_T_p[:, 0 + N_tau], F_Values[:, 0]) + 
                                    torch.dot(dFdy_T_p[:, 1 + N_tau], F_Values[:, 1]));

        # dL_dTheta
        for i in range(N_F_Params):
            dL_dTheta[i] -= 0.5*dt*(dFdTheta_T_p_t0[i] + dFdTheta_T_p_tjp1[i]);
        
        # dL_dPhi
        for i in range(N_X0_Params):
            dL_dPhi[i] -= 0.5*dt*(dX0dPhi_T_dFdy_T_p[i][..., 0] + dX0dPhi_T_dFdy_T_p[i][..., 1]);

        # Add the (dX0_dPhi(0)^T p(0) part of the dL_dPhi computation 
        for i in range(N_X0_Params):
            dL_dPhi[i] -= dX0dPhi_T_p_t0[i];



        ###########################################################################################
        # Debug plots!
        
        if(Debug_Plots == True):
            # Plot the final trajectory, gradients.
            plt.figure(0);
            plt.plot(t_Values, p[0, :].detach().numpy());
            plt.xlabel(r"$t$");
            plt.ylabel(r"$p(t)$");
            plt.title("Adjoint");

            plt.figure(1);
            plt.plot(t_Values, x_Pred_Values[0, :].detach().numpy());
            plt.xlabel(r"$t$");
            plt.ylabel("predicted position");
            plt.title("Predicted trajectory");
            plt.yscale('log');

            plt.figure(2);
            plt.plot(t_Values, dFdx_T_p[0, :].reshape(-1).detach().numpy());
            plt.xlabel(r"$t$");
            plt.ylabel(r"$(dF/dx)(t)$");
            plt.title("Gradient of F with respect to $x(t)$ along the predicted trajectory");

            plt.figure(3);
            plt.plot(t_Values, dFdy_T_p[0, :].reshape(-1).detach().numpy());
            plt.xlabel(r"$t$");
            plt.ylabel(r"$(dF/dy)(t)$");
            plt.title("Gradient of F with respect to $y(t) = x(t - tau)$ along the predicted trajectory");


        # All done... The kth return argument represents the gradient for the kth argument to forward.
        #      F,    X0,   tau,     T,    l,    G,    x_Targ_Interp, N_F_Params, N_X0_Params, Params 
        return None, None, dL_dtau, None, None, None, None,          None,       None,        *(dL_dTheta + dL_dPhi);
