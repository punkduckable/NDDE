from    typing  import Union; 

import  torch;


# Set up the logger.
import logging;
LOGGER : logging.Logger = logging.getLogger(__name__);



class L2_Cost(torch.nn.Module):
    def __init__(self, Weight : Union[float, torch.Tensor]) -> None:
        """
        A L2_Cost object is a functor which computes a weighted L2 norm between x and y. 
        Specifically, it computes \sum_{i = 1}^{n} Weight_i |x_i - y_i|^2.

        
        -------------------------------------------------------------------------------------------
        Arguments:

        Weight: This defines the weight in the weighted L2 norm. Weight can be one of three things:
        a float, a single element tensor, or a 1D tensor. If the weight is a float or a single 
        element tensor, we compute Weight*||x - y||_1. If it is a 1D tensor, then it must have the 
        same length as x, y. In this case, we compute sum_{i = 1}^{n} Weight_i |x_i - y_i|^2. 
        """

        # Run the super class initializer. 
        super(L2_Cost, self).__init__();
    
        # Make sure the Weight has the right type.
        if(  isinstance(Weight, float)):
            self.d      = None;
            self.Weight = torch.tensor([Weight], dtype = torch.float32);
            
        
        else:
            # Make sure we have a tensor.
            assert(isinstance(Weight, torch.Tensor));

            # We need to handle the single element tensor and 1D tensor cases separately. 
            if(  Weight.size == 1):
                self.d      = None;
                self.Weight = Weight.reshape(-1);
            
            
            else:
                assert(len(Weight.shape) == 1);
    
                self.d      = Weight.numel();
                self.Weight = Weight;



    def forward(self, x : torch.Tensor, y : torch.Tensor) -> torch.Tensor:
        """
        This function computes a weighted L2 norm squared between x and y. Thus, if x, y \in 
        \mathbb{R}^d, then we return 
                Weight_0*(x_0 - y_0)^2 + ... + Weight_{d - 1}*(x_{d - 1} - y_{d - 1})^2

                
        -----------------------------------------------------------------------------------------------
        Arguments:

        x, y: 1D tensors. They must have the same number of components.
        """

        # Run checks.
        assert(len(x.shape) == 1);
        assert(x.shape      == y.shape);

        # If self.Weight is a 1D vector, then the length of that vector must match the length of x
        # and y. If self.Weight is a single element Weight, or a float, then x and y can have 
        # arbitrary length; we set Weight_0 = ... = Weight_{d - 1} = self.Weight.
        if(self.d is not None):
            assert(x.numel() == self.d);
            return torch.dot(self.Weight, torch.square(x - y));
        else:
            return self.Weight*torch.sum(torch.square(x - y));



class L1_Cost(torch.nn.Module):
    def __init__(self, Weight : float = 1.0) -> None:
        """
        This class computes a weighted L1 squared norm between x and y. Specifically, it 
        computes Weight*||x(T) - y(T)||_1.
        """

        # Call the super class initializer.
        super(L1_Cost, self).__init__();

        # Store the weight.
        self.Weight = Weight;



    def forward(self, x : torch.Tensor, y : torch.Tensor) -> torch.Tensor:
        """ 
        Implements the terminal cost or "G" portion of the loss function for the NDDE algorithm. 
        Specifically, if x and y are in \mathbb{R}^d, then we return
            Weight*[ (x0 - y0)^2 + ... + (x_{d - 1} - y_{d - 1})^2 ]

        
        -----------------------------------------------------------------------------------------------
        Arguments:
    
        x, y: 1D tensors. They must have the same number of components.
        """

        return self.Weight*torch.sum(torch.abs(x - y));



def Integral_Loss(
            Predict_Trajectory      : torch.Tensor, 
            Target_Trajectory       : torch.Tensor,
            t_Trajectory            : torch.Tensor,
            l                       : torch.nn.Module) -> torch.Tensor:
    """
    This function approximates the loss 
        L(x_p(t), x_t(t)) = \int_{0}^{T} l(x_p(t), x_t(t)) dt
    Here, x_p represents the predicted trajectory while x_t is the true or target one. Likewise,
    l(x, y) is the function defined above. This could be an function like
        l(x, y) = ||x - y||^2 dt

    
    -----------------------------------------------------------------------------------------------
    Arguments: 

    Predict_Trajectory: The trajectory that we get when we use our current model to forecast the
    trajectory. This should be a N + 1 x d tensor, where N is the number of time steps. The i'th 
    row should, therefore, hold the value of the predicted solution at the i'th time value.

    Target_Trajectory: The trajectory we want the predicted trajectory to match. This should be 
    a N + 1 x d tensor whose i'th row holds the value of the true/target trajectory at the i'th 
    time value.

    t_Trajectory: a 1D tensor whose i'th element holds the time value associated with the i'th 
    row of the Predicted or Target trajectory. 
    
    l: The running cost function in the loss function above.

    
    -----------------------------------------------------------------------------------------------
    Returns:

    If there are N time steps, then we return the value
        \sum_{i = 0}^{N} 0.5*(t[i + 1] - t[i])*(l(x_p(t_i), x_t(t_i)) + l(x_p(t_{i + 1}), x_t(t_{i + 1})))
    where t_i represents the i'th entry of t_trajectory.
    """
    
    # Run checks!
    assert(len(Predict_Trajectory.shape)    == 2);
    assert(Predict_Trajectory.shape         == Target_Trajectory.shape)
    assert(Predict_Trajectory.shape[0]      == t_Trajectory.shape[0]);

    # Fetch number of data points. 
    d : int = Predict_Trajectory.shape[0];
    N : int = Predict_Trajectory.shape[1];

    # First compute the integrand.
    Integrand   : torch.Tensor  = torch.zeros(N);
    for j in range(N):
        Integrand[j] = l(Predict_Trajectory[j, :], Target_Trajectory[j, :]);

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
    trajectory. This should be a N + 1 x d tensor, where N is the number of time steps. The i'th 
    row should, therefore, hold the value of the predicted solution at the i'th time value.

    Target_Trajectory: The trajectory we want the predicted trajectory to match. This should be 
    a N + 1 x d tensor whose i'th row holds the value of the true/target trajectory at the i'th 
    time value.

    t_Trajectory: a 1D tensor whose i'th element holds the time value associated with the i'th 
    row of the Predicted or Target trajectory. We do not use this argument in this function.

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