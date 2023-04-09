import  torch;
import  numpy;
from    scipy               import  interpolate;



def Interpolate_Trajectory(x_trajectory : torch.Tensor, t_trajectory : torch.Tensor, N_Interp : int):
    """ 
    This function allows us to interpolate a vector valued function, x(t). We assume that the 
    kth column of X represents the value of x at the time t_trajectory[k]. This function 
    interpolates those samples to determine the value of x at 0, dt, d2t, ... , 
    t_trajectory[-1], where dt = t_trajectory[-1]/N.

    --------------------------------------------------------------------------------------------
    Arguments:

    x_trajectory: This is a d x N tensor whose kth column represents the value of some (vector 
    valued) function, x(t), at the kth time value. 

    t_trajectory: This is a N element tensor whose kth entry holds the kth time value.

    N_Interp: The number of time values at which we want the interpolated solution.
    """

    # Checks
    assert(len(x_trajectory.shape) == 2);
    assert(len(t_trajectory.shape) == 1);
    assert(x_trajectory.shape[1]   == t_trajectory.shape[0]);

    # Convert everything to numpy ndarrays.
    x_trajectory_np : numpy.ndarray = x_trajectory.detach().numpy();
    t_trajectory_np : numpy.ndarray = t_trajectory.detach().numpy();
    
    # interpolate
    f_interp = interpolate.interp1d(t_trajectory_np, x_trajectory_np);
    
    # Fetch the final time. 
    t_i : float = t_trajectory_np[0];
    T   : float = t_trajectory_np[-1];

    # find the values of the trajectory at the new time steps
    new_t_trajectory_np         : numpy.ndarray = numpy.linspace(t_i, T, N_Interp);
    interpolated_x_trajectory   : numpy.ndarray = f_interp(new_t_trajectory_np);

    return torch.from_numpy(interpolated_x_trajectory);