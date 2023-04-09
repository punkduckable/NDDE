import  torch;
from    typing  import Tuple;



def Forward_Euler(F : torch.nn.Module, x_0 : torch.Tensor, tau : torch.Tensor, T : torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    This function computes an approximate solution to the following DDE:
        x'(t)   = F(x(t), x(t - \tau), t)   t \in [0, T]
        x(t)    = x_0                       t \in [-\tau, 0]
    Here, x \in \mathbb{R}^d.
    
    --------------------------------------------------------------------------------------------
    Arguments:

    F : This is a torch.nn.Module object which represents the right-hand side of the DDE (See 
    above). 

    x_0 : This is a 1D tensor whose value represents the initial state of the DDE (see above).

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
    assert(len(x_0.shape)   == 1);
    assert(tau.numel()      == 1);
    assert(T.numel()        == 1);

    # Find the dimension of x. 
    d   : int           = x_0.shape[0];

    # Define the time-step to be 0.01 of the delay
    dt  : float         = 0.01*tau.item() if tau != 0 else 1.0;

    # Find the number of time steps. We add +1 for the initial time.
    N   : int           = int(torch.floor(T/dt).item());

    # compute the difference in indices between x(t) and x(t - tau).
    # This is just tau/dt = tau/(.01*tau) = 100.
    dN  : int           = 100;

    # tensor to hold the solution, time steps. Note the +1 is to account for 
    # the fact that we want the solution at N+1 times: 0, dt, 2dt, ... , Ndt.
    x_trajectory    : torch.Tensor  = torch.empty([d, N + 1],   dtype = torch.float32);
    t_trajectory    : torch.Tensor  = torch.linspace(start = 0, end = N*dt, steps = N + 1);

    # Set the first column of x to the IC.  
    x_trajectory[:, 0]             = x_0;
    
    # Compute the solution!
    for i in range(0, N):
        # Find x at the i+1th time value. Note that if t < tau (equivalently, i < dN), then 
        # t - \tau < 0, which means that x(t - \tau) = x_0. 
        if i >= dN:
            x_trajectory[:, i + 1] = x_trajectory[:, i] + dt*F(x = x_trajectory[:, i], y = x_trajectory[:, i - dN], t = torch.tensor(float(i*dN)));
        else:
            x_trajectory[:, i + 1] = x_trajectory[:, i] + dt*F(x = x_trajectory[:, i], y = x_0,          t = torch.tensor(float(i*dN)));

    # All done!
    return (x_trajectory, t_trajectory);