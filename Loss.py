from    typing  import Callable;

import  torch;


# Set up the logger.
import logging;
LOGGER : logging.Logger = logging.getLogger(__name__);


def l(x : torch.Tensor, y : torch.Tensor) -> torch.Tensor:
    """
    This function computes the L2 norm squared between x and y. Thus, if x, y \in R^d, then we 
    return 
            (x_0 - y_0)^2 + ... + (x_{d - 1} - y_{d - 1})^2

    -----------------------------------------------------------------------------------------------
    Arguments:

    x, y: 1D tensors. They must have the same number of components.
    """

    # Run checks.
    assert(len(x.shape) == 1);
    assert(x.shape      == y.shape);

    # Compute the L2 norm squared between x and y, return it.
    return torch.sum(torch.square(x - y));



def G(xT_Predict    : torch.Tensor, xT_Target) -> torch.Tensor:
    """ 
    Implements the "G" portion of the loss function for the NDDE algorithm.
    """

    return torch.sum(torch.square(xT_Predict - xT_Target));



def Integral_Loss(
            Predict_Trajectory      : torch.Tensor, 
            Target_Trajectory       : torch.Tensor,
            t_Trajectory            : torch.Tensor) -> torch.Tensor:
    """
    This function approximates the loss 
        L(x_p(t), x_t(t)) = \int_{0}^{T} l(x_p(t), x_t(t)) dt
    Here, x_p represents the predicted trajectory while x_t is the true or target one. Likewise,
    l(x, y) is the function defined above. This could be an function like
        l(x, y) = ||x - y||^2 dt

    -----------------------------------------------------------------------------------------------
    Arguments: 

    Predict_Trajectory: The trajectory that we get when we use our current model to forecast the
    trajectory. This should be a d x N + 1 tensor, where N is the number of time steps. The jth 
    column should, therefore, hold the value of the predicted solution at the jth time value.

    Target_Trajectory: The trajectory we want the predicted trajectory to match. This should be 
    a d x N + 1 tensor whose jth column holds the value of the true/target trajectory at the jth 
    time value.

    t_Trajectory: a 1D tensor whose jth element holds the time value associated with the jth column 
    of the Predicted or Target trajectory. 
    
    This function approximates the loss 
        L(x_p(t), x_t(t)) = \int_{0}^{T} l(x_p(t), x_t(t)) dt
    Here, x_p represents the predicted trajectory while x_t is the true or target one. 
    
    -----------------------------------------------------------------------------------------------
    Returns:

    If there are N time steps, then we return the value
        \sum_{j = 0}^{N} 0.5*(t[j + 1] - t[j])*(l(x_p(t_j), x_t(t_j)) + l(x_p(t_{j + 1}), x_t(t_{j + 1})))
    where t_j represents the jth entry of t_trajectory.
    """
    
    # Run checks!
    assert(len(Predict_Trajectory.shape)    == 2);
    assert(Predict_Trajectory.shape         == Target_Trajectory.shape)
    assert(Predict_Trajectory.shape[1]      == t_Trajectory.shape[0]);

    # Fetch number of data points. 
    d : int = Predict_Trajectory.shape[0];
    N : int = Predict_Trajectory.shape[1];

    # First compute the integrand.
    Integrand   : torch.Tensor  = torch.zeros(N);
    for j in range(N):
        Integrand[j] = l(Predict_Trajectory[:, j], Target_Trajectory[:, j]);

    # Now compute the loss using the trapezoidal rule.
    Loss        : torch.Tensor  = torch.zeros(1, dtype = torch.float32, requires_grad = True);
    for j in range(N - 1):
        Loss = Loss + 0.5*(t_Trajectory[j + 1] - t_Trajectory[j])*(Integrand[j] + Integrand[j + 1]);
    return Loss;



def SSE_Loss(   Predict_Trajectory  : torch.Tensor, 
                Target_Trajectory   : torch.Tensor,  
                t_Trajectory        : torch.Tensor) -> torch.Tensor:
    """
    This function defines the sum of squares error loss. In particular, for each time value, t_i, 
    we compute the square L2 loss between the predicted and target trajectories at time t_i. We 
    then sum these values across the time steps and return the resulting sum. 

    -----------------------------------------------------------------------------------------------
    Arguments:

    Precict_Trajectory: The trajectory that we get when we use our current model to forecast the
    trajectory. This should be a d x N + 1 tensor, where N is the number of time steps. The jth 
    column should, therefore, hold the value of the predicted solution at the jth time value.

    Target_Trajectory: The trajectory we want the predicted trajectory to match. This should be 
    a d x N + 1 tensor whose jth column holds the value of the true/target trajectory at the jth 
    time value.

    t_Trajectory: a 1D tensor whose jth element holds the time value associated with the jth column 
    of the Predicted or Target trajectory. We do not use this argument in this function.

    -----------------------------------------------------------------------------------------------
    Returns: 

    If there are N time steps, then we return 
        \sum_{i = 0}^{N} ||x_predict(t_i) - x_target(t_i)||_2^2.
    """

    # Run checks. 
    assert(len(Predict_Trajectory.shape)    == 2);
    assert(Predict_Trajectory.shape         == Target_Trajectory.shape);

    # Compute the loss. 
    Loss : torch.Tensor = torch.sum(torch.square(Predict_Trajectory - Target_Trajectory));
    return Loss;