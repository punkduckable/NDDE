import  torch; 
from    typing          import  List;

# Logger setup 
import logging;
LOGGER : logging.Logger = logging.getLogger(__name__);



class Logistic(torch.nn.Module):
    """ 
    Objects of the Logistic class are designed to model the right-hand side of an 
    exponential type DDE. Consider the following DDE:
        x'(t) = 
        x'(t) = F(x(t), x(t - \tau), t, \theta)     t \in [0, T]
        x(t)  = x_0                                 t \in [-\tau, 0]
    A Logistic object is supposed to act like the function F in the expression above 
    when F has the following general form 
        F(x(t), x(t - Model\tau), t, \theta) = \theta_0 x(t)(1 - \theta_1 x(t - \tau)).
    In other words, objects of this class are callable objects which accept three arguments: 
    x, y, and t. If x = x(t), y = x(t - \tau) then return F evaluated at those inputs. 
    """

    def __init__(self, theta_0 : float, theta_1 : float):
        """ 
        This class is set up to act as a logistic map:
            F(x, y, t) = c_0*x*(1 - c_1*y)
        (there is no explicit dependence on t). Thus, the arguments theta_0 and theta_1 define F.
        


        -------------------------------------------------------------------------------------------
        Arguments: 
        -------------------------------------------------------------------------------------------

        theta_0, theta_1: These should be floats representing the initial values of the variables 
        theta_0 and theta_1 in the definition above, respectively. We convert these to 
        torch.nn.Parameter objects which can be trained.
        """
        
        # Call the super class initializer. 
        super(Logistic, self).__init__();

        # Run checks.
        assert(isinstance(theta_0, float));
        assert(isinstance(theta_1, float));

        # Set model parameters.
        self.theta = torch.nn.parameter.Parameter(torch.tensor([theta_0, theta_1], dtype = torch.float32, requires_grad = True));



    def forward(self, x : torch.Tensor, y : torch.Tensor, t : torch.Tensor) -> torch.Tensor:
        """
        We expect x and y to be 1D tensors with the same number of elements. We also expect to be a
        single element tensor. This function then returns
            F(x, y, t) = theta_0*x*(1 - theta_1*y).
        
        
        
        -------------------------------------------------------------------------------------------
        Arguments:
        -------------------------------------------------------------------------------------------

        x, y: 1D tensors representing the first two arguments of the right hand side of the above 
        DDE. These should have the same length.

        t: a single element tensor whose lone value represents the third argument to the DDE above.
        """

        # Checks.
        assert(len(x.shape) == 1);
        assert(len(y.shape) == 1);
        assert(x.numel()    == y.numel());
        assert(x.numel()    == 1);
        assert(t.numel()    == 1);
        
        # compute, return the output         
        Output : torch.Tensor = self.theta[0]*x*(1. - self.theta[1]*y);
        return Output;




class Exponential(torch.nn.Module):
    """ 
    Objects of the Exponential class are designed to model the right-hand side of an 
    exponential type DDE. Consider the following DDE:
        x'(t) = 
        x'(t) = F(x(t), x(t - \tau), t, \theta)     t \in [0, T]
        x(t)  = x_0                                 t \in [-\tau, 0]
    A Exponential object is supposed to act like the function F in the expression above 
    when F has the following general form 
        F(x(t), x(t - \tau), t, \theta) = \theta_0 x(t) + \theta_1 x(t - \tau).
    In other words, objects of this class are callable objects which accept three arguments: 
    x, y, and t. If x = x(t), y = x(t - \tau) then return F evaluated at those inputs. 
    """

    def __init__(self, theta_0 : float, theta_1 : float):
        """ 
        This class defines the following right-hand side of a DDE:
            F(x, y, t) = theta_0*x + theta_1*y
        (there is no explicit dependence on t). Thus, the arguments theta_0 and theta_1 define F.

        

        -------------------------------------------------------------------------------------------
        Arguments: 
        -------------------------------------------------------------------------------------------

        theta_0, theta_1: These should be floats representing the initial values of the variables 
        theta_0 and theta_1 in the definition above, respectively. We convert these to 
        torch.nn.Parameter objects which can be trained.
        """
        
        # Call the super class initializer. 
        super(Exponential, self).__init__();

        # Run checks.
        assert(isinstance(theta_0, float));
        assert(isinstance(theta_1, float));

        # Set model parameters.
        self.theta    = torch.nn.parameter.Parameter(torch.tensor([theta_0, theta_1], dtype = torch.float32, requires_grad = True).reshape(-1));
        #self.theta_1    = torch.nn.parameter.Parameter(torch.tensor(theta_1, dtype = torch.float32, requires_grad = True).reshape(-1));


    def forward(self, x : torch.Tensor, y : torch.Tensor, t : torch.Tensor) -> torch.Tensor:
        """
        We expect x and y to be 1D tensors (with the same length). We also expect t to be a single 
        element tensor. This function then returns
            F(x, y, t) = \theta_0 x(t) + \theta_1 x(t - \tau).
        
            

        --------------------------------------------------------------------------------------------
        Arguments: 
        --------------------------------------------------------------------------------------------
        
        x, y: 1D tensors representing the first two arguments of the right hand side of the above 
        DDE. These should have the same length.

        t: a single element tensor whose lone value represents the third argument to the DDE above.
        """

        # Checks.
        assert(len(x.shape) == 1);
        assert(len(y.shape) == 1);
        assert(x.numel()    == y.numel());
        
        # compute, return the output         
        Output : torch.Tensor = self.theta[0]*x + self.theta[1]*y;
        return Output;



class Neural(torch.nn.Module):
    """ 
    A Neural class object is a neural network object which represents the right-hand side of a DDE.
    Consider the following DDE:
        x'(t) = 
        x'(t) = F(x(t), x(t - \tau), t, \theta)     t \in [0, T]
        x(t)  = x_0                                 t \in [-\tau, 0]
    A Neural object is supposed to act like the function F in the expression above when F is a 
    neural network. 

    In other words, objects of this class are callable objects which accept three arguments: 
    x, y, and t. If x = x(t), y = x(t - \tau) then return F evaluated at those inputs. 
    If x(t) \in \mathbb{R}^d, then the neural network should take inputs in \mathbb{R}^(2d + 1)
    and map to \mathbb{R}^d.
    """

    def __init__(self, Widths : List[int]):
        """ 
        This class defines the right-hand side of a DDE as a neural network. We use Widths to 
        define the widths of the layers in the network. We also use the softplus activation 
        function after each hidden layer    .


        
        --------------------------------------------------------------------------------------------
        Arguments:
        --------------------------------------------------------------------------------------------

        Widths: This should be a list of N + 1 integers, where N is the number of layers in the 
        neural network. Widths[0] represents the dimension of the domain, while Widths[-1] 
        represents the dimension of the co-domain. For i \in {1, 2, ... , N - 1}, Widths[i] 
        represents the width of the i'th hidden layer. Because a Neural object takes in x(t), y(t), 
        and t as inputs (and the former two live in \mathbb{R}^d), Widths[0] must be 2d + 1 (odd 
        and >= 3). Finally, Widths[1] must be d. 
        """
        
        # Call the super class initializer. 
        super(Neural, self).__init__();

        # Make sure Widths is a list of ints.
        self.N_Layers = len(Widths) - 1;
        for i in range(self.N_Layers + 1):
            assert(isinstance(Widths[i], int));
        
        # Find d, make sure 2*Widths[-1] + 1 == Widths[0].
        self.d = Widths[-1];
        assert(Widths[0] == 2*self.d + 1);
        self.Widths = Widths;

        # Set up the network's layers.
        self.Layers     = torch.nn.ModuleList();
        for i in range(self.N_Layers):
            self.Layers.append(torch.nn.Linear(in_features = Widths[i], out_features = Widths[i + 1]));
            torch.nn.init.xavier_normal_(self.Layers[i].weight);
            torch.nn.init.zeros_(self.Layers[i].bias);
        
        # Finally, set the activation function.
        self.Activation = torch.nn.Softplus();



    def forward(self, x : torch.Tensor, y : torch.Tensor, t : torch.Tensor) -> torch.Tensor:
        """
        This function passes x, y, and t through the neural network and returns F(x, y, t) (see 
        class doc string).
        
        --------------------------------------------------------------------------------------------
        Arguments:

        x, y: 1D tensors representing the first two arguments of the right hand side of the above 
        DDE. These should have the same length, d. 

        t: a single element tensor whose lone value represents the third argument to the DDE above.
        """

        # Checks.
        assert(len(x.shape) == 1);
        assert(len(y.shape) == 1);
        assert(x.numel()    == y.numel());
        assert(x.numel()    == self.d);
        assert(t.numel()    == 1);
        
        # Set up the input to the network.
        X : torch.Tensor = torch.concat([x.reshape(-1), y.reshape(-1), t.reshape(-1)], dim = 0);

        # Compute, return the output.  
        for i in range(self.N_Layers - 1):
            X = self.Activation(self.Layers[i](X));
        Output : torch.Tensor = self.Layers[-1](X);
        return Output;
