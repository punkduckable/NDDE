import  torch;
from    typing  import Tuple;



def Forward_Euler(F : torch.nn.Module, X0 : torch.Tensor, tau : torch.Tensor, T : torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    This function computes an approximate solution to the following DDE:
        x'(t)   = F(x(t), x(t - \tau), t)   t \in [0, T]
        x(t)    = X0(t)                     t \in [-\tau, 0]
    Here, x \in \mathbb{R}^d.
    
    --------------------------------------------------------------------------------------------
    Arguments:

    F : This is a torch.nn.Module object which represents the right-hand side of the DDE (See 
    above). 

    X0: This is a module which gives the initial condition in [-\tau, 0]. In particular, at
    time t \in [-\tau, 0], we set the initial condition at time t to X0(t). X0 should be a
    torch.nn.Module object which takes a tensor (of shape S) of time values and returns a S x d
    tensor[s, :] element holds the value of the IC at the s'th element of the input tensor.

    tau : This is a single element tensor whose lone element represents the time delay.

    T : this is a single element tensor whose lone element represents the final time. 

    --------------------------------------------------------------------------------------------
    Returns:

    A two element tuple. The first element holds a 2D tensor whose kth column holds the state of 
    the system (as solved by the solver) at the kth time value. The second is another 1D tensor 
    whose kth element holds the kth time value. Note: We only approximate the solution in the 
    interval [0, T].
    """

    # Checks
    assert(tau.numel()      == 1);
    assert(tau.item()       >  0);
    assert(T.numel()        == 1);

    # Define the time-step to be 0.01 of the delay
    dt  : float         = 0.1*tau.item() if tau != 0 else 1.0;

    # Find the number of time steps. We add +1 for the initial time.
    N   : int           = int(torch.floor(T/dt).item());

    # compute the difference in indices between x(t) and x(t - tau).
    # This is just tau/dt = tau/(.1*tau) = 10.
    N_tau       : int           = 10;

    # Pre-compute X0 at -tau, -tau + dt, ... , 0.
    t_Values_X0 : torch.Tensor  = torch.linspace(start = -tau.item(), end = 0, steps = N_tau + 1);
    X0_t        : torch.Tensor  = X0(t_Values_X0);

    # Find the dimension of x. Note that X0_t should be a (N_tau + 1) x d tensor, so we can use it
    # to find d. 
    d           : int           = X0_t.shape[1];

    # tensor to hold the solution, time steps. Note the +1 is to account for 
    # the fact that we want the solution at N+1 times: 0, dt, 2dt, ... , Ndt.
    x_Trajectory    : torch.Tensor  = torch.empty([d, N + 1],   dtype = torch.float32);
    t_Trajectory    : torch.Tensor  = torch.linspace(start = 0, end = T, steps = N + 1);

    # Set the first column of x to the IC at time 0 (the last element of t_Values_X0).
    x_Trajectory[:, 0]             = X0_t[-1];
    
    # Compute the solution!
    for i in range(0, N):
        # Find x at the i+1th time value. Note that if t < tau (equivalently, i < N_tau), then 
        # t - \tau < 0, which means that x(t - \tau) = x0. 
        if i >= N_tau:
            x_Trajectory[:, i + 1] = x_Trajectory[:, i] + dt*F(x = x_Trajectory[:, i], y = x_Trajectory[:, i - N_tau], t = torch.tensor(float(i*N_tau)));
        else:
            x_Trajectory[:, i + 1] = x_Trajectory[:, i] + dt*F(x = x_Trajectory[:, i], y = X0_t[i, :],                  t = torch.tensor(float(i*N_tau)));

    # All done!
    return (x_Trajectory, t_Trajectory);



def RK2(F : torch.nn.Module, X0 : torch.Tensor, tau : torch.Tensor, T : torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    This function computes an approximate solution to the following DDE:
        x'(t)   = F(x(t), x(t - \tau), t)   t \in [0, T]
        x(t)    = X0(t)                     t \in [-\tau, 0]
    Here, x \in \mathbb{R}^d.
    
    --------------------------------------------------------------------------------------------
    Arguments:

    F : This is a torch.nn.Module object which represents the right-hand side of the DDE (See 
    above). 

    X0: This is a module which gives the initial condition in [-\tau, 0]. In particular, at
    time t \in [-\tau, 0], we set the initial condition at time t to X0(t). X0 should be a
    torch.nn.Module object which takes a tensor (of shape S) of time values and returns a S x d
    tensor[s, :] element holds the value of the IC at the s'th element of the input tensor.

    tau : This is a single element tensor whose lone element represents the time delay.

    T : this is a single element tensor whose lone element represents the final time. 

    --------------------------------------------------------------------------------------------
    Returns:

    A two element tuple. The first element holds a 2D tensor whose kth column holds the state of 
    the system (as solved by the solver) at the kth time value. The second is another 1D tensor 
    whose kth element holds the kth time value. Note: We only approximate the solution in the 
    interval [0, T].
    """

    # Checks
    assert(tau.numel()      == 1);
    assert(tau.item()       >  0);
    assert(T.numel()        == 1);

    # Define the time-step to be 0.01 of the delay
    dt  : float         = 0.1*tau.item() if tau != 0 else 1.0;

    # Find the number of time steps. We add +1 for the initial time.
    N   : int           = int(torch.floor(T/dt).item());

    # compute the difference in indices between x(t) and x(t - tau).
    # This is just tau/dt = tau/(.1*tau) = 10.
    N_tau       : int           = 10;

    # Pre-compute X0 at -tau, -tau + dt, ... , 0.
    t_Values_X0 : torch.Tensor  = torch.linspace(start = -tau.item(), end = 0, steps = N_tau + 1);
    X0_t        : torch.Tensor  = X0(t_Values_X0);

    # Find the dimension of x. Note that X0_t should be a (N_tau + 1) x d tensor, so we can use it
    # to find d. 
    d           : int           = X0_t.shape[1];

    # tensor to hold the solution, time steps. Note the +1 is to account for 
    # the fact that we want the solution at N+1 times: 0, dt, 2dt, ... , Ndt.
    x_Trajectory    : torch.Tensor  = torch.empty([d, N + 1],   dtype = torch.float32);
    t_Trajectory    : torch.Tensor  = torch.linspace(start = 0, end = T, steps = N + 1);

    # Set the first column of x to the IC at time 0 (the last element of t_Values_X0).
    x_Trajectory[:, 0]             = X0_t[-1, :];
    
    # Compute the solution!
    for i in range(0, N):
        # Find x at the i+1th time value. We do this using a 2 step RK method. Note that if t < tau 
        # (equivalently, i < N_tau), then t - \tau < 0, which means that x(t - \tau) = x0. 
        t_i     : torch.Tensor  = t_Trajectory[i];                                                      # t
        x_i     : torch.Tensor  = x_Trajectory[:, i];                                                   # x(t)
        y_i     : torch.Tensor  = x_Trajectory[:, i - N_tau] if i > N_tau else X0_t[i, :];              # x(t - tau)
        k1      : torch.Tensor  = F(x_i, y_i, t_i);

        t_ip1   : torch.Tensor  = t_Trajectory[i + 1];                                                  # t + dt
        x_ip1   : torch.Tensor  = x_i + dt*k1;                                                          # x(t) + dt*k1
        y_ip1   : torch.Tensor  = x_Trajectory[:, i + 1 - N_tau] if i + 1 > N_tau else X0_t[i + 1, :];  # x(t + dt - tau)
        k2      : torch.Tensor  = F(x_ip1, y_ip1, t_ip1);

        x_Trajectory[:, i + 1]  = x_Trajectory[:, i] + dt*0.5*(k1 + k2);

    # All done!
    return (x_Trajectory, t_Trajectory);