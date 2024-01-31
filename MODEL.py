import torch; 

# Logger setup 
import logging;
LOGGER : logging.Logger = logging.getLogger(__name__);



class Logistic_Model(torch.nn.Module):
    """ 
    Objects of the Exponential_Model class are designed to model the right-hand side of an 
    exponential type DDE. Consider the following DDE:
        x'(t) = 
        x'(t) = F(x(t), x(t - \tau), t, \theta)     t \in [0, T]
        x(t)  = x_0                                 t \in [-\tau, 0]
    A Exponential_Model object is supposed to act like the function F in the expression above 
    when F has the following general form 
        F(x(t), x(t - Model\tau), t, \theta) = \theta_0 x(t)(1 - \theta_1 x(t - \tau)).
    In other words, objects of this class are callable objects which accept three arguments: 
    x, y, and t. If x = x(t), y = x(t - \tau) then return F evaluated at those inputs. 
    """

    def __init__(self, theta_0 : float = 2.0, theta_1 : float = 2.0):
        """ 
        Currently, this MODEL class is set up to act as a logistic map:
            F(x, y, t) = c_0*x*(1 - c_1*y)
        (there is no explicit dependence on t). Thus, the arguments theta_0 and theta_1 define the 
        function  implied by this MODEL object.
        """
        
        # Call the super class initializer. 
        super(Logistic_Model, self).__init__();

        # Set model parameters.
        self.Params = torch.nn.parameter.Parameter(torch.tensor([theta_0, theta_1], dtype = torch.float32, requires_grad = True));



    def forward(self, x : torch.Tensor, y : torch.Tensor, t : torch.Tensor) -> torch.Tensor:
        """
        A MODEL object defines the right hand side of a DDE (see class doc-string above). Thus, the 
        forward map defines that function. We expect x, y, and t to be single element tensors. This 
        function then returns
            F(x, y, t) = theta_0*x*(1 - theta_1*y)
        
        --------------------------------------------------------------------------------------------
        Arguments:

        x, y: 1D tensors representing the first two arguments of the right hand side of the above 
        DDE. These should have the same length.

        t: a single element tensor whose lone value represents the third argument to the DDE above.
        """

        # Checks.
        assert(len(x.shape) == 1);
        assert(len(y.shape) == 1);
        assert(x.numel()    == y.numel());
        assert(t.numel()    == 1);
        
        # compute, return the output         
        Output : torch.Tensor = self.Params[0]*x*(1. - self.Params[1]*y);
        return Output;




class Exponential_Model(torch.nn.Module):
    """ 
    Objects of the Exponential_Model class are designed to model the right-hand side of an 
    exponential type DDE. Consider the following DDE:
        x'(t) = 
        x'(t) = F(x(t), x(t - \tau), t, \theta)     t \in [0, T]
        x(t)  = x_0                                 t \in [-\tau, 0]
    A Exponential_Model object is supposed to act like the function F in the expression above 
    when F has the following general form 
        F(x(t), x(t - \tau), t, \theta) = \theta_0 x(t) + \theta_1 x(t - \tau).
    In other words, objects of this class are callable objects which accept three arguments: 
    x, y, and t. If x = x(t), y = x(t - \tau) then return F evaluated at those inputs. 
    """

    def __init__(self, theta_0 : float = -2.0, theta_1 : float = -2.0):
        """ 
        Currently, this MODEL class is set up to act as a logistic map:
            F(x, y, t) = theta_0*x + theta_1*y
        (there is no explicit dependence on t). Thus, the arguments theta_0 and theta_1 define the function 
        implied by this MODEL object.
        """
        
        # Call the super class initializer. 
        super(Exponential_Model, self).__init__();

        # Set model parameters.
        self.Params = torch.nn.parameter.Parameter(torch.tensor([theta_0, theta_1], dtype = torch.float32, requires_grad = True));



    def forward(self, x : torch.Tensor, y : torch.Tensor, t : torch.Tensor) -> torch.Tensor:
        """
        A MODEL object defines the right hand side of a DDE (see class doc-string above). Thus, the 
        forward map defines that function. We expect x, y, and t to be single element tensors. This 
        function then returns
            F(x, y, t) = theta_0*x*(1 - theta_1*y)
        
        --------------------------------------------------------------------------------------------
        Arguments:

        x, y: 1D tensors representing the first two arguments of the right hand side of the above 
        DDE. These should have the same length.

        t: a single element tensor whose lone value represents the third argument to the DDE above.
        """

        # Checks.
        assert(len(x.shape) == 1);
        assert(len(y.shape) == 1);
        assert(x.numel()    == y.numel());
        assert(t.numel()    == 1);
        
        # compute, return the output         
        Output : torch.Tensor = self.Params[0]*x + self.Params[1]*y;
        return Output;