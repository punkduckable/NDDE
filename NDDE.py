import  torch; 
from    typing  import  Tuple, Callable;
from    scipy   import  interpolate;
import  matplotlib.pyplot as plt;

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
        

    def forward(self, x_0 : torch.Tensor, tau : torch.Tensor, T : torch.Tensor, l : Callable, x_True_Interp : Callable):
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

        x_True_Interp: An interpolation object for the true trajectory. We need this to be able 
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
        #Trajectory = DDE_adjoint_SSE.apply(Model, x_0, tau, T, Model_Params);
        Trajectory = DDE_adjoint_l.apply(Model, x_0, tau, T, l, x_True_Interp, Model_Params);
        return Trajectory;



class DDE_adjoint_SSE(torch.autograd.Function):
    """
    This class implements the forward and backward passes for updating the parameters and tau. This
    particular class is designed for the SSE loss function. 
    
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
                F           : torch.nn.Module, 
                x_0         : torch.Tensor, 
                tau         : torch.Tensor, 
                T           : torch.Tensor, 
                F_Params    : torch.Tensor) -> torch.Tensor:
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
        ctx.F               = F; 

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
        dt              : float             = 0.1*tau.item();
        N               : int               = int(torch.floor(T/dt).item());
        N_tau           : int               = 10;

        # Now, let's set up an interpolation of the forward trajectory. We will evaluate this 
        # interpolation at each time when we want to compute the adjoint. This allows us to use
        # a different time step for the forward and backwards passes. 
        x_Pred_Interp                       = interpolate.interp1d(x = t_Trajectory.detach().numpy(), y = x_Trajectory.detach().numpy())

        # Find time values for backwards pass. 
        t_Values        : torch.Tensor      = torch.linspace(start = 0, end = T, steps = N + 1);

        # evaluate the interpolation of the predicted, true solution at these values. 
        x_Pred_Values   : torch.Tensor      = torch.from_numpy(x_Pred_Interp(t_Values.detach().numpy()));

        # Set up a tensor to hold the adjoint. p[i, :, j] holds the value of the ith adjoint at the
        # jth time value.    
        p               : torch.Tensor      = torch.zeros([d, d, N + 1],    dtype = torch.float32);

        # Initialize the last component of p_i to be e_i.
        for i in range(d):
            e_i : torch.Tensor = torch.zeros(d);
            e_i[i] = 1.0;
            p[i, :, -1]                         = e_i;

        # we need the sensitivity matrix evaluated at future times in the equation for the adjoint
        # so we need to store it
        F_Values    = torch.zeros([d,           N + 1], dtype = torch.float32);
        dF_dx       = torch.zeros([d, d,        N + 1], dtype = torch.float32);
        dF_dy       = torch.zeros([d, d,        N + 1], dtype = torch.float32);
        dF_dtheta   = torch.zeros([d, N_Params, N + 1], dtype = torch.float32);



        ###########################################################################################
        # Compute dF_dx, dF_dy, dF_dtheta, and p at each time step.

        # Populate the initial elements of dF_dy and dF_dx. Since we use an RK2 method to solve the 
        # p adjoint equation, we always need to know dF_dx and dF_dy one step ahead of the current 
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

            # Fetch x, y from the i-1th time step.
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
                
                p[i, :, j - 1] = p[i, :, j] - dt*0.5*(k1 + k2);
            """
            for i in range(d):
                if j >= N - N_tau:
                    p[i, :, j - 1] = p[i, :, j] - dt*(-torch.mv(torch.transpose(dF_dx[:, :, j], 0, 1), p[i, :, j])); 
                else:
                    p[i, :, j - 1] = p[i, :, j] - dt*(-torch.mv(torch.transpose(dF_dx[:, :, j], 0, 1), p[i, :, j]) + 
                                                      -torch.mv(torch.transpose(dF_dy[:, :, j + N_tau], 0, 1), p[i, :, j + N_tau]));
            """

        # Plot the final trajectory, gradients.
        plt.figure(0);
        plt.plot(t_Values, p[0, 0, :].detach().numpy());
        plt.xlabel("$t$");
        plt.ylabel("$p(t)$");
        plt.title("Adjoint");

        plt.figure(1);
        plt.plot(t_Values, x_Pred_Values[0, :].detach().numpy());
        plt.xlabel("$t$");
        plt.ylabel("predicted position");
        plt.title("Predicted trajectory");
        plt.yscale('log');

        plt.figure(2);
        plt.plot(t_Values, dF_dx.reshape(-1).detach().numpy());
        plt.xlabel("$t$");
        plt.ylabel("$dF/dx(t)$");
        plt.title("Gradient of F with respect to $x(t)$ along the predicted trajectory");

        plt.figure(3);
        plt.plot(t_Values, dF_dy.reshape(-1).detach().numpy());
        plt.xlabel("$t$");
        plt.ylabel("$dF/dy(t)$");
        plt.title("Gradient of F with respect to $y(t) = x(t - tau)$ along the predicted trajectory");


    
        ###########################################################################################
        # Compute dx_dtheta and dx_dtau at each time step.

        dx_dtheta   = torch.zeros([d, N_Params, N + 1], dtype = torch.float32);
        dx_dtau     = torch.zeros([d, N + 1],            dtype = torch.float32);

        for i in range(0, d):
            for j in range(0, N):
                # From the paper, 
                #   dx^i(t_j)/dtheta = \int_0^t_j p_i(T - t_j + t) (dF/dtheta)(x(t), x(t - tau)) dt
                # Here, p_i is the adjoint for the ith component of p. We compute this integral 
                # the trapezodial rule.
                for k in range(0, j):
                    dx_dtheta[i, :, j] += dt*0.5*(  torch.matmul(p[i, :, N - j + k    ].reshape(1, -1), dF_dtheta[:, :, k    ]).reshape(-1) + 
                                                    torch.matmul(p[i, :, N - j + k + 1].reshape(1, -1), dF_dtheta[:, :, k + 1]).reshape(-1) );
                
                # From the paper, 
                #   dx^i(t_j)/dtau  -= \int_0^{t_j - tau} p_i(T - t_j + t + tau) (dF/dtau)(x(t + tau), x(t), t + tau)F(x(t), x(t - tau), t) dt
                # We also compute this integral using the trapezodial rule.
                for k in range(0, j - N_tau):
                    dx_dtau[i, j] -= dt*0.5*(   torch.dot(p[i, :, N - j + k + N_tau    ],   torch.mv(dF_dy[:, :, k + N_tau    ], F_Values[:, k    ])) + 
                                                torch.dot(p[i, :, N - j + k + N_tau + 1],   torch.mv(dF_dy[:, :, k + N_tau + 1], F_Values[:, k + 1])) );



        ###########################################################################################
        # Compute dL_dtau and dL_dtheta.
        
        # Set up vectors to hold dL_dtau and dL_dtheta
        dL_dtau         : torch.Tensor      = torch.tensor([0.], dtype = torch.float32);
        dL_dtheta       : torch.Tensor      = torch.zeros_like(F_Params);

        # Now compute dL_dtheta and dL_dtau. In this case, 
        #   dL_dtau = \sum_{j = 0}^{N_data - 1} \sum_{i = 0}^{d} -2*(x_true_i(t_j) - x_pred_i(t_j))*dx_i(t_j)/dtau
        #           = \sum_{j = 0}^{N_data - 1} (dL/dx_pred(t_j))*dx(t_j)/dtau
        # Crucially, we know that the jth column of grad_y holds dL/dx_pred(t_j) (think about it). 
        # Analogously, 
        #   dL_dtheta = \sum_{j = 0}^{N_data - 1} (dL/dx_pred(t_j))*dx(t_j)/dtheta
        for j in range(0, N + 1):
            dL_dtheta   += torch.matmul(grad_y[:, j].reshape(1, -1), dx_dtheta[:, :, j]).reshape(-1);
            dL_dtau     += torch.dot(   grad_y[:, j],                dx_dtau  [:, j   ]);



        # All done... The kth return argument represents the gradient for the kth argument to forward.
        return None, p[0], dL_dtau, None, dL_dtheta;




class DDE_adjoint_l(torch.autograd.Function):
    """
    This class implements the forward and backward passes for updating the parameters and tau. This
    particular class is designed for a loss function of the form
        \int_{0}^{T} l(x_Predict, x_True) dx 

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
                x_True_Interp   : Callable, 
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
            Loss(x_Pred) = \int_{0}^{T} l(x_Predict(t), x_Target(t)) dt
        Thus, it should be a Callable object which takes two arguments, both in R^d. We assume that
        this function can be differentiated (using autograd) with respect to its first argument.

        x_True_Interp: An interpolation object for the true trajectory. We need this to be able 
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
        ctx.x_True_Interp   = x_True_Interp;
        ctx.l               = l;

        # Save tensor arguments for backwards
        ctx.save_for_backward(x_0, tau, T, x_Trajectory, t_Trajectory, F_Params);
        
        # All done!
        return x_Trajectory.clone();
        
    

    @staticmethod
    def backward(ctx, grad_y : torch.Tensor) -> Tuple[torch.Tensor]:
        # recover information from the forward pass
        F               : torch.nn.Module                   = ctx.F;
        x_True_Interp   : Callable                          = ctx.x_True_Interp;
        l               : Callable                          = ctx.l;
        x_0, tau, T, x_Trajectory, t_Trajectory, F_Params   = ctx.saved_tensors;
        d               : int                               = x_0.numel();

        # extract the parameters from the list and find how many there are
        F_Params        : torch.Tensor      = F.Params;
        N_Params        : int               = F_Params.shape[0];
        
        # Find the step size for the backwards pass. Also find the number of time step and the 
        # number of time steps in an interval of length tau. 
        dt              : float             = 0.1*tau.item();
        N               : int               = int(torch.floor(T/dt).item());
        N_tau           : int               = 10;

        # Now, let's set up an interpolation of the forward trajectory. We will evaluate this 
        # interpolation at each time when we want to compute the adjoint. This allows us to use
        # a different time step for the forward and backwards passes. 
        x_Pred_Interp                       = interpolate.interp1d(x = t_Trajectory.detach().numpy(), y = x_Trajectory.detach().numpy())

        # Find time values for backwards pass. 
        t_Values        : torch.Tensor      = torch.linspace(start = 0, end = T, steps = N + 1);

        # evaluate the interpolation of the predicted, true solution at these values. 
        x_True_Values   : torch.Tensor      = torch.from_numpy(x_True_Interp(t_Values.detach().numpy()));
        x_Pred_Values   : torch.Tensor      = torch.from_numpy(x_Pred_Interp(t_Values.detach().numpy()));

        # Set up a tensor to hold the adjoint. p[:, j] holds the value of the adjoint at the
        # jth time value.   
        # Notice that starting with a tensor of zeros actually initializes the adjoint at the final
        # time. From the paper, the adjoint at time T should be set to dg/dx(T). However, since we 
        # assume that there is no g, the IC is just 0. 
        p               : torch.Tensor      = torch.zeros([d, N + 1],    dtype = torch.float32);

        # Set up vectors to track F, dF_dx, dF_dy, dF_dtheta, and dl_dx at each time. This is 
        # helpful for debugging purposes (and storing this information is necessary to compute 
        # some quantities).
        F_Values    = torch.zeros([d,           N + 1], dtype = torch.float32);
        dF_dx       = torch.zeros([d, d,        N + 1], dtype = torch.float32);
        dF_dy       = torch.zeros([d, d,        N + 1], dtype = torch.float32);
        dF_dtheta   = torch.zeros([d, N_Params, N + 1], dtype = torch.float32);
        dl_dx       = torch.zeros([d,           N + 1], dtype = torch.float32);
        


        ###########################################################################################
        # Compute dF_dx, dF_dy, dF_dtheta, and p at each time step.

        # Populate the initial elements of dF_dy and dF_dx. Since we use an RK2 method to solve the 
        # p adjoint equation, we always need to know dF_dx and dF_dy one step ahead of the current 
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

        # Compute dl_dx at the final time step.
        x_True_N    : torch.Tensor  = x_True_Values[:, N];
        l_x_N       : torch.Tensor  = l(x_N, x_True_N);
        dl_dx[:, -1]                = torch.autograd.grad(outputs = l_x_N, inputs = x_N)[0];

        # We are all done tracking gradients.
        torch.set_grad_enabled(False);


        for j in range(N, 0, -1):  
            # Enable gradient tracking! We need this to compute the gradients of F. 
            torch.set_grad_enabled(True);

            # Fetch x, y from the i-1th time step.
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

            # Compute dl_dx at the j-1'th time step.
            x_True_jm1  : torch.Tensor  = x_True_Values[:, j - 1];
            l_x_jm1     : torch.Tensor  = l(x_jm1, x_True_jm1);
            dl_dx[:, j - 1]             = torch.autograd.grad(outputs = l_x_jm1, inputs = x_jm1)[0];

            #print("%8.5f, %8.5f, " % (dl_dx[:, j].item(), (2*(x_Pred_Values[:, j] - x_True_Values[:, j])).item()), end = ' ');

            # We are all done tracking gradients.
            torch.set_grad_enabled(False);
        
            # Find p at the previous time step. Recall that p satisfies the following DDE:
            #       p'(t) = -dF_dx(t)^T p(t)  - dF_dy(t)^T p(t + tau) 1_{t + tau < T}(t) + d l(x(t))/d x_pred(t)
            #       p(t)  = 0               if t > T    
            # Note: since p(t) = 0 for t > T, the delay term vanishes if t + tau > T. We find a 
            # numerical solution to this DDE using the RK2 method. In this case, we compute
            #       p(t - dt) \approx p(t) - dt*0.5*(k1 + k2)
            #       k1 = F(t_i, x(t_i), x(t_i - tau))
            #       k2 = F(t_i - dt, z(t) - dt*k1, z(t - dt + tau))
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
                p[:, j - 1] = p[:, j] - dt*(-torch.mv(torch.transpose(dF_dx[:, :, j        ], 0, 1), p[:, j        ]) + 
                                            dl_dx[:, j]);
            else: 
                p[:, j - 1] = p[:, j] - dt*(-torch.mv(torch.transpose(dF_dx[:, :, j        ], 0, 1), p[:, j        ]) + 
                                            -torch.mv(torch.transpose(dF_dy[:, :, j + N_tau], 0, 1), p[:, j + N_tau]) + 
                                            dl_dx[:, j]);

        # Plot the final trajectory, gradients
        plt.figure(0);
        plt.plot(t_Values, p[0, :].detach().numpy());
        plt.xlabel("$t$");
        plt.ylabel("$p^0(t)$");
        plt.title("Adjoint");

        plt.figure(1);
        plt.plot(t_Values, x_Pred_Values[0, :].detach().numpy());
        plt.xlabel("$t$");
        plt.ylabel("predicted position");
        plt.title("Predicted trajectory");
        plt.yscale('log');

        plt.figure(2);
        plt.plot(t_Values, dF_dx.reshape(-1).detach().numpy());
        plt.xlabel("$t$");
        plt.ylabel("$dF/dx(t)$");
        plt.title("Gradient of F with respect to $x(t)$ along the predicted trajectory");

        plt.figure(3);
        plt.plot(t_Values, dF_dy.reshape(-1).detach().numpy());
        plt.xlabel("$t$");
        plt.ylabel("$dF/dy(t)$");
        plt.title("Gradient of F with respect to $y(t) = x(t - tau)$ along the predicted trajectory");

        plt.figure(4);
        plt.plot(t_Values, dl_dx.reshape(-1).detach().numpy());
        plt.xlabel("$t$");
        plt.ylabel("$dl/dx(t)$");
        plt.title("Gradient of l with respect to $x(t)$ along the predicted trajectory");



        ###########################################################################################
        # Compute dL_dtau and dL_dtheta.
        
        # Set up vectors to hold dL_dtau and dL_dtheta
        dL_dtheta       : torch.Tensor      = torch.zeros_like(F_Params);
        dL_dtau         : torch.Tensor      = torch.tensor([0.], dtype = torch.float32);

        # Now compute dL_dtheta and dL_dtau. In this case, 
        #   dL_dtheta   =  \int_{t = 0}^T p(t) dF_dtheta(x(t), x(t - tau), t) dt
        #   dL_dtau     = -\int_{t = 0}^{T - tau} p(t + tau) dF_dy(x(t + tau), x(t), t) F(x(t), x(t - tau), t) dt
        # We compute these integrals using the trapezodial rule.
        for j in range(0, N):
            dL_dtheta   -=  0.5*dt*(torch.matmul(p[:, j    ].reshape(1, -1), dF_dtheta[:, :, j    ]).reshape(-1) +
                                    torch.matmul(p[:, j + 1].reshape(1, -1), dF_dtheta[:, :, j + 1]).reshape(-1));
        for j in range(N_tau, N):
            dL_dtau     +=  0.5*dt*(torch.dot(p[:, j    ], torch.mv(dF_dy[:, :, j    ], F_Values[:, j - N_tau    ])) + 
                                    torch.dot(p[:, j + 1], torch.mv(dF_dy[:, :, j + 1], F_Values[:, j - N_tau + 1])));

        # All done... The kth return argument represents the gradient for the kth argument to forward.
        return None, p[0], dL_dtau, None, None, None, dL_dtheta;
