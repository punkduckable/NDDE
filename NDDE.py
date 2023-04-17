import  torch; 
from    typing  import  Tuple;
from    scipy   import  interpolate;

from    Solver  import  RK2         as DDE_Solver;



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
        

    def forward(self, x_0 : torch.Tensor, tau : torch.Tensor, T : torch.Tensor, x_True_Interp):
        """
        Arguments: 

        x_0: the initial position. We assume that x(t) = x_0 for t \in [-\tau, 0]. Thus, x_0 
        defines the initial state of the DDE. x_0 should be a 1D tensor.

        tau: The time delay in the DDE. This should be a single element tensor.

        T: The final time in our solution to the DDE (See above). This should be a single 
        element tensor.

        x_True_Interp: An interpolation object we can use to evaluate the true solution at 
        various points in time. We need this during the backward step.
        """

        # Run checks.
        assert(len(x_0.shape)   == 1);
        assert(tau.numel()      == 1);
        assert(T.numel()        == 1);

        # Fetch the model and its parameters.
        Model           : torch.nn.Module   = self.Model;
        Model_Params    : torch.Tensor      = Model.Params;

        # Evaluate the neural DDE using the Model
        Trajectory = DDE_adjoint_1D.apply(Model, x_0, tau, T, x_True_Interp, Model_Params);
        return Trajectory;



class DDE_adjoint_1D(torch.autograd.Function):
    """
    This function implements the adjoint method so that we can compute the gradients of the loss 
    with respect to tau and the Model's parameters. This class defines a forward and backwards 
    pass.

    Forward pass - solve the DDE with f(x, y, theta) as the vector field and x_0 as initial 
    condition.

    Backward pass - Solve the adjoint equation to return the gradient of the loss with respect to
    theta and tau. 
    """

    @staticmethod
    def forward(ctx, F : torch.nn.Module, x_0 : torch.Tensor, tau : torch.Tensor, T : torch.Tensor, x_True_Interp, F_Params : torch.Tensor) -> torch.Tensor:
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

        x_True_Interp: An interpolation object we can use to evaluate the true solution at 
        various points in time. We need this during the backward step.

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
        ctx.x_True_Interp   = x_True_Interp;

        # Save tensor arguments for backwards
        ctx.save_for_backward(x_0, tau, T, x_Trajectory, t_Trajectory, F_Params);
        
        # All done!
        return x_Trajectory.clone();
        
    

    @staticmethod
    def backward(ctx, grad_y : torch.Tensor) -> Tuple[torch.Tensor]:
        # recover information from the forward pass
        F               : torch.nn.Module                   = ctx.F;
        x_True_Interp                                       = ctx.x_True_Interp;
        x_0, tau, T, x_Trajectory, t_Trajectory, F_Params   = ctx.saved_tensors;
        d               : int                               = x_0.numel();

        # extract the parameters from the list and find how many there are
        F_Params        : torch.Tensor      = F.Params;
        N_Params        : int               = F_Params.shape[0];
        
        # Find the step size for the backwards pass. Also find the number of time step and the 
        # number of time steps in an interval of length tau. 
        dt              : float             = 0.01*tau.item();
        N               : int               = int(torch.floor(T/dt).item());
        N_tau           : int               = 100;

        # Now, let's set up an interpolation of the forward trajectory. We will evaluate this 
        # interpolation at each time when we want to compute the adjoint. This allows us to use
        # a different time step for the forward and backwards passes. 
        x_Pred_Interp                       = interpolate.interp1d(x = t_Trajectory.detach().numpy(), y = x_Trajectory.detach().numpy())

        # Find time values for backwards pass. 
        t_Values        : torch.Tensor      = torch.linspace(start = 0, end = T, steps = N + 1);

        # Evaluate the interpolation of the true solution at these values.
        x_True_Values   : torch.Tensor      = torch.from_numpy(x_True_Interp(t_Values.detach().numpy()));

        # evaluate the interpolation of the predicted, true solution at these values. 
        x_Pred_Values   : torch.Tensor      = torch.from_numpy(x_Pred_Interp(t_Values.detach().numpy()));

        # Set up a vector to hold the adjoint. The ith component of p will hold the value of p(t) 
        # at the ith time step.   
        p               : torch.Tensor      = torch.zeros([d, d, N + 1],    dtype = torch.float32);

        # Initialize the last component of p. 
        p[:, :, -1]                         = 1.;

        # we need the sensitivity matrix evaluated at future times in the equation for the adjoint
        # so we need to store it
        F_Values    = torch.zeros([d,           N + 1], dtype = torch.float32);
        dF_dx       = torch.zeros([d, d,        N + 1], dtype = torch.float32);
        dF_dy       = torch.zeros([d, d,        N + 1], dtype = torch.float32);
        dF_dtheta   = torch.zeros([d, N_Params, N + 1], dtype = torch.float32);



        ###########################################################################################
        # Compute dF_dx, dF_dy, dF_dtheta, and p at each time step.

        # Populate the initial elements of df_dy and df_dx. Since we use an RK2 method to solve the 
        # p adjoint equation, we always need to know df_dx and df_dy one step ahead of the current 
        # time step. 
        torch.set_grad_enabled(True);

        # Fetch t, x, y from the last time step.
        t_N : torch.Tensor  = t_Values[N];
        x_N : torch.Tensor  = x_Pred_Values[:, N         ].requires_grad_(True);
        y_N : torch.Tensor  = x_Pred_Values[:, N - N_tau ].requires_grad_(True);

        # Evaluate F at the current state.
        F_N : torch.Tensor  = F(x_N, y_N, t_N);
        F_Values[:, -1]     = F_N;

        # find the gradient of F_i with respect to x, y, and theta at the final time step.
        for i in range(d):
            dFi_dx_N, dFi_dy_N, dFi_dtheta_N    = torch.autograd.grad(outputs = F_N[i], inputs = (x_N, y_N, F_Params));
            dF_dx[i, :,  -1]                    = dFi_dx_N;
            dF_dy[i, :, -1]                     = dFi_dy_N;
            dF_dtheta[i, :, -1]                 = dFi_dtheta_N;

        # We are all done tracking gradients.
        torch.set_grad_enabled(False);

        for j in range(N, 0, -1):  
            # Enable gradient tracking! We need this to compute the gradients of F. 
            torch.set_grad_enabled(True);

            # Fetch x, y from the ith, i-1th time step.
            t_j         : torch.Tensor  = t_Values[j];
            x_j         : torch.Tensor  = x_Pred_Values[:, j].requires_grad_(True);
            y_j         : torch.Tensor  = x_Pred_Values[:, j - N_tau ].requires_grad_(True) if j - N_tau >= 0 else x_0;

            t_jm1       : torch.Tensor  = t_Values[j - 1];
            x_jm1       : torch.Tensor  = x_Pred_Values[:, j - 1         ].requires_grad_(True);
            y_jm1       : torch.Tensor  = x_Pred_Values[:, j - 1 - N_tau ].requires_grad_(True) if j - 1 - N_tau >= 0 else x_0;

            # Evaluate F at the i-1th time step.
            F_jm1       : torch.Tensor  = F(x_jm1, y_jm1, t_jm1);
            F_Values[:, j - 1]          = F_jm1;

            # find the gradient of F_i with respect to x, y, and theta at the i-1th time step.
            for i in range(d):
                dFi_dx_jm1, dFi_dy_jm1, dFi_dtheta_jm1 = torch.autograd.grad(outputs = F_jm1[i], inputs = (x_jm1, y_jm1, F_Params));
                
                dF_dx[i, :, j - 1]     = dFi_dx_jm1;
                dF_dy[i, :, j - 1]     = dFi_dy_jm1; 
                dF_dtheta[i, :, j - 1] = dFi_dtheta_jm1;

            # We are all done tracking gradients.
            torch.set_grad_enabled(False);
        
            # Find p at the previous time step. Recall that p_i satisfies the following DDE:
            #       p_i'(t) = -dF_dx(t)^T p_i(t)  - dF_dy(t)^T p_i(t + tau) 1_{t + tau < T}(t)
            #       p_i(t)  = 0               if t > T    
            # Note: since p_i(t) = 0 for t > T, the delay term vanishes if t + tau > T. We find a 
            # numerical solution to this DDE using the RK2 method. In this case, we compute
            #       p(t - dt) \approx p(t) - dt*0.5*(k1 + k2)
            #       k1 = F(t_i, x(t_i), x(t_i - tau))
            #       k2 = F(t_i - dt, z(t) - dt*k1, z(t - dt + tau))
            for i in range(d):
                k1 : torch.Tensor   = -torch.mv(torch.transpose(dF_dx[:, :, j], 0, 1), p[i, :, j]);
                if(j + N_tau < N):
                    k1 += -torch.mv(torch.transpose(dF_dy[:, :, j + N_tau], 0, 1), p[i, :, j + N_tau]);
                
                k2 : torch.Tensor   = -torch.mv(torch.transpose(dF_dx[:, :, j - 1], 0, 1), (p[i, :, j] - dt*k1));
                if(j - 1 + N_tau < N):
                    k2 += -torch.mv(torch.transpose(dF_dy[:, :, j - 1 + N_tau], 0, 1), p[i, :, j - 1 + N_tau]);
                
                p[i, :, j - 1] = p[i, :, j] + dt*0.5*(k1 + k2);
    

        """
        ###########################################################################################
        # Compute dx_dtheta and dx_dtau at each time step.

        dx_dtheta   = torch.zeros([d, N_Params, N + 1], dtype = torch.float32);
        dx_dtau     = torch.zeros([d, N + 1],            dtype = torch.float32);

        for i in range(d):
            for j in range(N + 1):
                # From the paper, 
                #   dx^i(t_j)/dtheta = \int_0^t_j p_i(T - t_j + t) (dF/dtheta)(x(t), x(t - tau)) dt
                # Here, p_i is the adjoint for the ith component of p. We compute this integral 
                # the trapezodial rule.
                for k in range(j):
                    dx_dtheta[i, :, j] += dt*0.5*(p[i, N - j + k]*dF_dtheta[:, k] + p[i, N - j + k + 1]*dF_dtheta[:, k + 1]);
                
                # From the paper, 
                #   dx^i(t_j)/dtau  -= \int_0^{t_j - tau} p_i(T - t_j + t + tah) (dF/dtheta)(x(t + tau), x(t), t + tau)F(x(t), x(t - tau), t) dt
                # We also compute this integral using the trapezodial rule.
                for k in range(j - N_tau):
                    dx_dtau[i, j] -= dt*0.5*p[i, N + j + k + N_tau]*dF_dy[:, k + N_tau]*F_Values[:, k];


        ###########################################################################################
        # Compute dL_dtau and dL_dtheta.
        
        # Set up vectors to hold dL_dtau and dL_dtheta
        dL_dtau         : torch.Tensor      = torch.tensor([0.], dtype = torch.float32);
        dL_dtheta       : torch.Tensor      = torch.zeros_like(F_Params);

        for j in range(0, N + 1):
            for i in range(d):
                dL_dtheta   += -2*(x_True_Values[i, j] - x_Pred_Values[i, j])*dx_dtheta[i, :, j];
                dL_dtau     += -2*(x_True_Values[i, j] - x_Pred_Values[i, j])*dx_dtau[i, j];
        """

        # Set up vectors to hold dL_dtau and dL_dtheta
        dL_dtau         : torch.Tensor      = torch.tensor([0.], dtype = torch.float32);
        dL_dtheta       : torch.Tensor      = torch.zeros_like(F_Params);

        for j in range(0, N):
            # Get current time. 
            t_j     = t_Values[j];
            t_jp1   = t_Values[j + 1];

            # update the gradient for theta. I do this using the trapezodial rule.
            dL_dtheta : torch.Tensor = dL_dtheta - dt*0.5*(p[0, :, j]*dF_dtheta[0, :, j] + p[0, :, j + 1]*dF_dtheta[0, :, j + 1]);

            # Update the gradient for tau. Note that the integral for dL/dtau goes from 
            # tau to T (after a change of variables). 
            if t_j.item() >= tau.item():
                x_j_tau   : torch.Tensor  = x_Pred_Values[:, j - N_tau];
                y_j_tau   : torch.Tensor  = x_Pred_Values[:, j - 2*N_tau] if j - 2*N_tau >= 0 else x_0;
                t_j_tau   : torch.Tensor  = t_j - tau;
                F_j_tau   : torch.Tensor  = F(x_j_tau, y_j_tau, t_j_tau);

                x_jp1_tau     : torch.Tensor  = x_Pred_Values[:, j + 1 - N_tau  ];
                y_jp1_tau     : torch.Tensor  = x_Pred_Values[:, j + 1 - 2*N_tau] if j + 1 - 2*N_tau >= 0 else x_0;
                t_jp1_tau     : torch.Tensor  = t_jp1 - tau;                
                F_jp1_tau     : torch.Tensor  = F(x_jp1_tau, y_jp1_tau, t_jp1_tau);

                dL_dtau      : torch.Tensor  = dL_dtau - dt*0.5*(p[0, :, j]*dF_dy[0, :, j]*F_j_tau + p[0, :, j + 1]*dF_dy[0, :, j + 1]*F_jp1_tau);

        # All done... The kth return argument represents the gradient for the kth argument to forward.
        return None, p[0].clone(), dL_dtau, None, None, dL_dtheta;