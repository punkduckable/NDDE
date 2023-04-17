import torch;



def L2_Loss(Predict_Trajectory      : torch.Tensor, 
            Target_Trajectory       : torch.Tensor, 
            t_Trajectory            : torch.Tensor) -> torch.Tensor:
    """
    TO DO

    This function approximates the loss 
        L(x_p(t), x_t(t)) = \int_{0}^{T} ||x_p(t) - x_t(t)||^2 dt
    Here, x_p represents the predicted trajectory while x_t is the true or target one. 
    """
    
    # Run checks!
    assert(len(Predict_Trajectory.shape)    == 2);
    assert(Predict_Trajectory.shape         == Target_Trajectory.shape)
    assert(Predict_Trajectory.shape[1]      == t_Trajectory.shape[0]);

    # Fetch number of data points. 
    N_Data : int = Predict_Trajectory.shape[1];

    # First compute the integrand, which we call the residual.
    Residual    : torch.Tensor  = torch.sum(torch.square(Predict_Trajectory - Target_Trajectory), dim = 0);

    # Now compute the loss using the trapezodial rule.
    Loss        : torch.Tensor  = torch.zeros(1, dtype = torch.float32);
    for i in range(N_Data - 1):
        Loss += (t_Trajectory[i + 1] - t_Trajectory[i])*0.5*(Residual[i + 1] + Residual[i]);
    return Loss;



def SSE_Loss(Predict_Trajectory : torch.Tensor, Target_Trajectory : torch.Tensor) -> torch.Tensor:
    """
    This function defines the sum of squares error loss. In particular, for each time value, t_i, 
    we compute the square L2 loss between the predicted and target trajectories at time t_i. We 
    then sum these values across the time steps and return the resulting sum. 

    -----------------------------------------------------------------------------------------------
    Arguments:

    Predicted trajectory: The trajectory that we get when we use our current model to forecast the
    trajectory.

    Target_Trajectory: The trajectory we want the predicted trajectory to match.

    -----------------------------------------------------------------------------------------------
    Returns: 

    If there are N time steps, then we return 
        \sum_{i = 0}^{N - 1} ||x_predict(t_i) - x_target(t_i)||_2^2.
    """

    # Run checks. 
    assert(len(Predict_Trajectory.shape)    == 2);
    assert(Predict_Trajectory.shape         == Target_Trajectory.shape);

    # Compute the loss. 
    Loss : torch.Tensor = torch.mean(torch.square(Predict_Trajectory - Target_Trajectory));
    return Loss;