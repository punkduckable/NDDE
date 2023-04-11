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
        

    def forward(self, x_0 : torch.Tensor, tau : torch.Tensor, T : torch.Tensor):
        """
        Arguments: 

        x_0: the initial position. We assume that x(t) = x_0 for t \in [-\tau, 0]. Thus, x_0 
        defines the initial state of the DDE. x_0 should be a 1D tensor.

        tau: The time delay in the DDE. This should be a single element tensor.

        T: The final time in our solution to the DDE (See above). This should be a single 
        element tensor.
        """

        # Run checks.
        assert(len(x_0.shape)   == 1);
        assert(tau.numel()      == 1);
        assert(T.numel()        == 1);

        # Fetch the model and its parameters.
        Model           : torch.nn.Module   = self.Model;
        Model_Params    : torch.Tensor      = Model.Params;

        # Evaluate the neural DDE using the Model
        Trajectory = DDE_adjoint_1D.apply(Model, x_0, tau, T, Model_Params);
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
    def forward(ctx, F : torch.nn.Module, x_0 : torch.Tensor, tau : torch.Tensor, T : torch.Tensor, F_Params : torch.Tensor) -> torch.Tensor:
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
        ctx.F       = F; 

        # Save tensor arguments for backwards
        ctx.save_for_backward(x_0, tau, T, x_Trajectory, t_Trajectory, F_Params);
        
        # All done!
        return x_Trajectory.clone();
        
    

    @staticmethod
    def backward(ctx, grad_y : torch.Tensor) -> Tuple[torch.Tensor]:
        # recover information from the forward pass
        F               : torch.nn.Module                   = ctx.F;
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
        x_Interp = interpolate.interp1d(x = t_Trajectory.detach().numpy(), y = x_Trajectory.detach().numpy())

        # Find time values for backwards pass. 
        t_Values     : torch.Tensor         = torch.linspace(start = 0, end = T, steps = N + 1);

        # evaluate the interpolation of x at these values. 
        x_Values        : torch.Tensor      = torch.from_numpy(x_Interp(t_Values.detach().numpy()));

        # define the augmented system
        p               : torch.Tensor      = torch.zeros([d, N + 1], dtype = torch.float32); # we need to remember the second dimension since the eq for p is a DDE
        p_tau           : torch.Tensor      = torch.tensor([0.], dtype = torch.float32);
        p_theta         : torch.Tensor      = torch.zeros_like(F_Params);

        # Initialize the last component of p. 
        p[:, -1]                            = 1.;

        # we need the sensitivity matrix evaluated at future times in the equation for the adjoint
        # so we need to store it
        df_dy = torch.zeros([d, N + 1], dtype = torch.float32);
        df_dx = torch.zeros([d, N + 1], dtype = torch.float32);

        # Populate the initial elements of df_dy and df_dx. Since we use an RK2 method to solve the 
        # p adjoint equation, we always need to know df_dx and df_dy one step ahead of the current time step. 
        torch.set_grad_enabled(True);

        # Fetch t, x, y from the last time step.
        t_N  : torch.Tensor = t_Values[N];
        x_N : torch.Tensor  = x_Values[:, N         ].requires_grad_(True);
        y_N : torch.Tensor  = x_Values[:, N - N_tau ].requires_grad_(True);

        # Evaluate F at the current state.
        F_N : torch.Tensor  = F(x_N, y_N, t_N);

        # find the gradient of F w.r.t. x and y.
        df_dx_N, df_dy_N, df_dtheta_i   = torch.autograd.grad(outputs = F_N, inputs = (x_N, y_N, F_Params));
        df_dx[:, -1]                    = df_dx_N;
        df_dy[:, -1]                    = df_dy_N;

        # We are all done tracking gradients.
        torch.set_grad_enabled(False);

        for i in range(N, 0, -1):  
            # Enable gradient tracking! We need this to compute the gradients of F. 
            torch.set_grad_enabled(True);

            # Fetch x, y from the ith time step.
            t_i     : torch.Tensor  = t_Values[i];
            t_im1   : torch.Tensor  = t_Values[i - 1];
            x_im1   : torch.Tensor  = x_Values[:, i - 1         ].requires_grad_(True);
            y_im1   : torch.Tensor  = x_Values[:, i - 1 - N_tau ].requires_grad_(True) if i - 1 - N_tau >= 0 else x_0;

            # Evaluate F at the current state.
            F_im1   : torch.Tensor  = F(x_im1, y_im1, t_im1);

            # find the gradient of F w.r.t. x, y, and theta
            df_dx_im1, df_dy_im1, df_dtheta_im1 = torch.autograd.grad(outputs = F_im1, inputs = (x_im1, y_im1, F_Params));

            # We are all done tracking gradients.
            torch.set_grad_enabled(False);

            # Check if the gradient with respect to x or y is None. If so, then F doesn't depend on 
            # that parameter. We set the gradient to zero and let the user know.
            if(df_dx_im1 == None):
                df_dx[:, i - 1] = torch.zeros(d);
                print("No dependence on x(t)!");
            else:
                df_dx[:, i - 1] = df_dx_im1;
            
            if(df_dy_im1 == None):
                df_dy[:, i - 1] = torch.zeros(d);
                print("No dependence on the delay term!");
            else:
                df_dy[:, i - 1] = df_dy_im1;
        
            # Find p at the previous time step. Recall that p satisfies the following DDE:
            #       p'(t) = -df_dx(t)^T p(t) - df_dy(t)^T p(t + tau) 1_{t + tau < T}(t)
            # since p(t) = 0 for t > T, the delay term vanishes if t + tau > T
            k1 : torch.Tensor   = -df_dx[:, i]*p[:, i];
            if(i + N_tau < N):
                k1 += -df_dy[:, i + N_tau]*p[:, i + N_tau];
            
            k2 : torch.Tensor   = -df_dx[:, i - 1]*(p[:, i] - dt*k1);
            if(i - 1 + N_tau < N):
                k2 += -df_dy[:, i - 1 + N_tau]*p[:, i - 1 + N_tau];
            
            p[:, i - 1] = p[:, i] + dt*0.5*(k1 + k2);

            """
            if i + N_tau >= N:
                p[i - 1] = p[i] - dt*(df_dx[:, i]*p[i]);
            else:
                p[i - 1] = p[i] - dt*(df_dx[:, i]*p[i] + df_dy[:, i + N_tau]*p[i + N_tau]);
            """
                
            # update the gradient for theta. I do this using the trapezodial rule.
            p_theta : torch.Tensor = p_theta - dt*0.5*(p[:, i - 1]*df_dtheta_im1 + p[:, i]*df_dtheta_i);

            # Update the gradient for tau. Note that the integral for dL/dtau goes from 
            # tau to T (after a change of variables). 
            if t_im1.item() >= tau.item():
                x_im1_tau   : torch.Tensor  = x_Values[:, i - 1 - N_tau];
                y_im1_tau   : torch.Tensor  = x_Values[:, i - 1 - 2*N_tau] if i - 1 - 2*N_tau >= 0 else x_0;
                t_im1_tau   : torch.Tensor  = t_im1 - tau;
                F_im1_tau   : torch.Tensor  = F(x_im1_tau, y_im1_tau, t_im1_tau);

                x_i_tau     : torch.Tensor  = x_Values[:, i - N_tau  ];
                y_i_tau     : torch.Tensor  = x_Values[:, i - 2*N_tau] if i - 2*N_tau >= 0 else x_0;
                t_i_tau     : torch.Tensor  = t_i - tau;                
                F_i_tau     : torch.Tensor  = F(x_i_tau, y_i_tau, t_i_tau);

                p_tau   : torch.Tensor = p_tau - dt*0.5*(p[:, i - 1]*df_dy[:, i - 1]*F_im1_tau + p[:, i]*df_dy[:, i]*F_i_tau);

            # Finally, update df_dtheta_i
            df_dtheta_i = df_dtheta_im1;

        """
        plt.figure(0);
        plt.plot(t_trajectory, p.detach().numpy());
        plt.yscale('log');
        plt.figure(1);
        plt.plot(t_trajectory, df_dx.reshape(-1).detach().numpy());
        plt.figure(2);
        print(df_dy);
        plt.plot(t_trajectory, df_dy.reshape(-1).detach().numpy());
        """
        
        # All done... The kth return argument represents the gradient for the kth argument to forward.
        return None, p[0].clone(), p_tau, None, p_theta;