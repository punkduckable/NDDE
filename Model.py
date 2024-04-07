import  torch; 
from    typing          import  List, Dict;

# Logger setup 
import logging;
LOGGER : logging.Logger = logging.getLogger(__name__);



class Logistic(torch.nn.Module):
    """ 
    Objects of the Logistic class are designed to model the right-hand side of an exponential type 
    DDE. Consider the following DDE:
        x'(t) = F(x(t), x(t - \tau), \tau, t, \theta)   t \in [0, T]
        x(t)  = X0(t)                                   t \in [-\tau, 0]
    A Logistic object is supposed to act like the function F in the expression above when F has the 
    following general form 
        F(x(t), x(t - \tau), \tau, t, \theta) = \theta_0 x(t)(1 - \theta_1 x(t - \tau)).
    In other words, objects of this class are callable objects which accept four arguments: x, y, 
    tau, and t. If x = x(t), y = x(t - \tau) then return F evaluated at those inputs. 
    """

    def __init__(self, theta_0 : float, theta_1 : float):
        """ 
        This class is set up to act as a logistic map:
            F(x, y, t) = c_0*x*(1 - c_1*y)
        (there is no explicit dependence on t or tau). Thus, the arguments theta_0 and theta_1 
        define F.
        


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



    def forward(self, x : torch.Tensor, y : torch.Tensor, tau : torch.Tensor, t : torch.Tensor) -> torch.Tensor:
        """
        We expect x and y to be 1D tensors with the same number of elements. We also expect to be a
        single element tensor. This function then returns
            F(x, y, tau, t) = theta_0*x*(1 - theta_1*y).
        (there is no explicit dependence on t or tau).

        
        
        -------------------------------------------------------------------------------------------
        Arguments:
        -------------------------------------------------------------------------------------------

        x, y: 1D tensors representing the first two arguments of the right hand side of the above 
        DDE. These should have the same length.

        tau: A single element tensor whose lone value represents the delay in the DDE (this only 
        matters if the delay appears explicitly in F).

        t: a single element tensor whose lone value represents the third argument to the DDE above.
        """

        # Checks.
        assert(len(x.shape) == 1);
        assert(len(y.shape) == 1);
        assert(x.numel()    == y.numel());
        assert(x.numel()    == 1);
        assert(t.numel()    == 1);
        assert(tau.numel()  == 1);
        
        # compute, return the output         
        Output : torch.Tensor = self.theta[0]*x*(1. - self.theta[1]*y);
        return Output;




class Exponential(torch.nn.Module):
    """ 
    Objects of the Exponential class are designed to model the right-hand side of an exponential 
    type DDE. Consider the following DDE:
        x'(t) = F(x(t), x(t - \tau), \tau, t, \theta)   t \in [0, T]
        x(t)  = X0(t)                                   t \in [-\tau, 0]
    A Exponential object is supposed to act like the function F in the expression above when F has the 
    following general form 
        F(x(t), x(t - \tau), \tau, t, \theta) = \theta_0 x(t)(1 - \theta_1 x(t - \tau)).
    In other words, objects of this class are callable objects which accept four arguments: x, y, 
    tau, and t. If x = x(t), y = x(t - \tau) then return F evaluated at those inputs. 
    """

    def __init__(self, theta_0 : float, theta_1 : float):
        """ 
        This class defines the following right-hand side of a DDE:
            F(x, y, tau, t) = theta_0*x + theta_1*y
        (there is no explicit dependence on t or tau). Thus, the arguments theta_0 and theta_1 
        define F.

        

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


    def forward(self, x : torch.Tensor, y : torch.Tensor, tau : torch.Tensor, t : torch.Tensor) -> torch.Tensor:
        """
        We expect x and y to be 1D tensors (with the same length). We also expect t to be a single 
        element tensor. This function then returns
            F(x, y, tau, t) = \theta_0 x + \theta_1 y
        (there is no explicit dependence on t or tau).


        --------------------------------------------------------------------------------------------
        Arguments: 
        --------------------------------------------------------------------------------------------
        
        x, y: 1D tensors representing the first two arguments of the right hand side of the above 
        DDE. These should have the same length.

        tau: A single element tensor whose lone value represents the delay in the DDE (this only 
        matters if the delay appears explicitly in F).

        t: a single element tensor whose lone value represents the third argument to the DDE above.
        """

        # Checks.
        assert(len(x.shape) == 1);
        assert(len(y.shape) == 1);
        assert(x.numel()    == y.numel());
        assert(t.numel()    == 1);
        assert(tau.numel()  == 1);

        # compute, return the output         
        Output : torch.Tensor = self.theta[0]*x + self.theta[1]*y;
        return Output;



class Cheyne(torch.nn.Module):
    """ 
    Objects of the Cheyne class are designed to model the right-hand side of a DDE of the
    following form:
        x'(t) = p - V0*x(t)[x(t - tau)^m]/[a + x^m(t - tau)]
        x(0)  = X0(t) 
    Here, p, V0, and a are parameters. This equation appears on page 22 of the book 
    "Mathematical Biology" by Murray. 
    """

    def __init__(self, p : float = 6.0, V0 : float = 7.0, a : float = 1.0, m: int = 10):
        """ 
        This class defines the following right-hand side of a DDE:
            F(x, y, tau, t) =  p - V0*x[y^m]/[a + y^m]
        (there is no explicit dependence on t or tau). Thus, the arguments theta_0 and theta_1 
        define F.

        

        -------------------------------------------------------------------------------------------
        Arguments: 
        -------------------------------------------------------------------------------------------

        p, V0, a: These should be floats representing the initial values of the variables p, V0, 
        and a in the definition above, respectively. We convert these to torch.nn.Parameter 
        objects which we can train.

        m: An integer representing the variable "m" in the definition above.
        """
        
        # Call the super class initializer. 
        super().__init__();

        # Run checks.
        assert(isinstance(p,  float));
        assert(isinstance(V0, float));
        assert(isinstance(a,  float));
        assert(isinstance(m,  int));

        # Set model parameters.
        self.p  = torch.nn.parameter.Parameter(torch.tensor([p],  dtype = torch.float32, requires_grad = True).reshape(-1));
        self.V0 = torch.nn.parameter.Parameter(torch.tensor([V0], dtype = torch.float32, requires_grad = True).reshape(-1));
        self.a  = torch.nn.parameter.Parameter(torch.tensor([a],  dtype = torch.float32, requires_grad = True).reshape(-1));
        self.m  = torch.tensor(m, dtype = torch.int32);



    def forward(self, x : torch.Tensor, y : torch.Tensor, tau : torch.Tensor, t : torch.Tensor) -> torch.Tensor:
        """
        We expect x and y to be 1D tensors (with the same length). We also expect t to be a single 
        element tensor. This function then returns
            F(x, y, tau, t) =  p - V0*x[y^m]/[a + y^m]
        (there is no explicit dependence on t or tau).
            

        --------------------------------------------------------------------------------------------
        Arguments: 
        --------------------------------------------------------------------------------------------
        
        x, y: 1D tensors representing the first two arguments of the right hand side of the above 
        DDE. These should have the same length.

        tau: A single element tensor whose lone value represents the delay in the DDE (this only 
        matters if the delay appears explicitly in F).

        t: a single element tensor whose lone value represents the third argument to the DDE above.
        """

        # Checks.
        assert(len(x.shape) == 1);
        assert(len(y.shape) == 1);
        assert(x.numel()    == y.numel());
        assert(x.numel()    == 1);
        assert(t.numel()    == 1);
        assert(tau.numel()  == 1);
        
        # compute, return the output         
        Output : torch.Tensor = self.p - self.V0*x*torch.pow(y, self.m)/(self.a + torch.pow(y, self.m));
        return Output;





class HIV(torch.nn.Module):
    """ 
    HIV class objects model the dynamics of infected cells and HIV viruses in hunan cells. This 
    involves the following DDE in \mathbb{R}^3, 
        (d/dt)T*(t)     = k T0 VI(t - tau) exp(-m tau) - d T*(t)
        (d/dt)V_{I}(t)  = (1 - np) d N T*(t) - c V_{I}(t)
        (d/dt)V_{NI}    = np d N T*(t) - c V_{NI}(t)
    Here, k, d, np, N, and c are learnable parameters. T0 is a fixed constant. This equation 
    appears is from the following paper:
        
        Nelson, Patrick W., James D. Murray, and Alan S. Perelson. "A model of HIV-1 pathogenesis 
        that includes an intracellular delay." Mathematical biosciences 163.2 (2000): 201-215.
    """

    def __init__(   self, 
                    k   : float = 0.0000343,  
                    T0  : float = 180.0, 
                    m   : float = 6.0, 
                    d   : float = 0.5, 
                    np  : float = 0.43, 
                    N   : float = 480.0, 
                    c   : float = 0.25):
        """ 
        This class defines the following right-hand side of a DDE:
                                { k T0 y[1] exp(-m tau) - d x[0]
            F(x, y, tau, t) =   { (1 - np) d N x[0] - c x[1]
                                { np d N x[0] - c x[2]
        (there is no explicit dependence on t). Thus, the arguments theta_0 and theta_1 define F.

        

        -------------------------------------------------------------------------------------------
        Arguments: 
        -------------------------------------------------------------------------------------------

        k, m, d, np, N, c: Trainable parameters in the above model.

        T0: The population of non-infected T-cells. This is not a trainable parameter.
        """
        
        # Call the super class initializer. 
        super().__init__();

        # Run checks.
        assert(isinstance(k,  float));
        assert(isinstance(d,  float));
        assert(isinstance(np, float));
        assert(isinstance(N,  float));
        assert(isinstance(c,  float));
        assert(isinstance(T0,  float));

        # Set model parameters.
        self.k  = torch.nn.parameter.Parameter(torch.tensor([k], dtype = torch.float32, requires_grad = True).reshape(-1));
        self.m  = torch.nn.parameter.Parameter(torch.tensor([m], dtype = torch.float32, requires_grad = True).reshape(-1));
        self.d  = torch.nn.parameter.Parameter(torch.tensor([d], dtype = torch.float32, requires_grad = True).reshape(-1));
        self.np = torch.nn.parameter.Parameter(torch.tensor([np], dtype = torch.float32, requires_grad = True).reshape(-1));
        self.N  = torch.nn.parameter.Parameter(torch.tensor([N], dtype = torch.float32, requires_grad = True).reshape(-1));
        self.c  = torch.nn.parameter.Parameter(torch.tensor([c], dtype = torch.float32, requires_grad = True).reshape(-1));
        
        self.T0 = torch.tensor([T0], dtype = torch.float32).reshape(-1);



    def forward(self, x : torch.Tensor, y : torch.Tensor, tau : torch.Tensor, t : torch.Tensor) -> torch.Tensor:
        """
        We expect x and y to be 1D tensors (with the same length). We also expect t to be a single 
        element tensor. This function then returns
            F(x, y, tau, t) =   { k T0 y[1] exp(-m tau) - d x[0]
                                { (1 - np) d N x[0] - c x[1]
                                { np d N x[0] - c x[2]
            

                                
        --------------------------------------------------------------------------------------------
        Arguments: 
        --------------------------------------------------------------------------------------------
        
        x, y: 1D tensors representing the first two arguments of the right hand side of the above 
        DDE. These should have the same length.

        tau: A single element tensor whose lone value represents the delay in the DDE (this only 
        matters if the delay appears explicitly in F).

        t: a single element tensor whose lone value represents the third argument to the DDE above.
        """

        # Checks.
        assert(len(x.shape) == 1);
        assert(len(y.shape) == 1);
        assert(x.numel()    == y.numel());
        assert(x.numel()    == 3);
        assert(t.numel()    == 1);
        assert(tau.numel()  == 1);

        # compute, return the output  torch.exp(-self.m*tau)
        F0  : torch.Tensor  = (self.k * self.T0 * torch.exp(-self.m*tau))*y[1]  - self.d * x[0];
        F1  : torch.Tensor  = ((1. - self.np) * self.d * self.N)*x[0]           - self.c * x[1];
        F2  : torch.Tensor  = (self.np        * self.d * self.N)*x[0]           - self.c * x[2];
        return torch.concat([F0, F1, F2]);



class Neural(torch.nn.Module):
    """ 
    A Neural class object is a neural network object which represents the right-hand side of a DDE.
    Consider the following DDE:
        x'(t) = F(x(t), x(t - \tau), tau, t, \theta)    t \in [0, T]
        x(t)  = x_0                                     t \in [-\tau, 0]
    A Neural object is supposed to act like the function F in the expression above when F is a 
    neural network. 

    In other words, objects of this class are callable objects which accept four arguments: 
    x, y, tau, and t. If x = x(t), y = x(t - \tau) then return F evaluated at those inputs. 
    If x(t) \in \mathbb{R}^d, then the neural network should take inputs in \mathbb{R}^(2d + 2)
    and map to \mathbb{R}^d.
    """

    def __init__(self, Widths : List[int]):
        """ 
        This class defines the right-hand side of a DDE as a neural network. We use Widths to 
        define the widths of the layers in the network. We also use the softplus activation 
        function after each hidden layer.


        
        --------------------------------------------------------------------------------------------
        Arguments:
        --------------------------------------------------------------------------------------------

        Widths: This should be a list of N + 1 integers, where N is the number of layers in the 
        neural network. Widths[0] represents the dimension of the domain, while Widths[-1] 
        represents the dimension of the co-domain. For i \in {1, 2, ... , N - 1}, Widths[i] 
        represents the width of the i'th hidden layer. Because a Neural object takes in x(t), y(t), 
        tau, and t as inputs (and the former two live in \mathbb{R}^d), Widths[0] must be 2d + 2 
        (even and >= 4). Finally, Widths[1] must be d. 
        """
        
        # Call the super class initializer. 
        super(Neural, self).__init__();

        # Make sure Widths is a list of ints.
        self.N_Layers = len(Widths) - 1;
        for i in range(self.N_Layers + 1):
            assert(isinstance(Widths[i], int));
        
        # Find d, make sure 2*Widths[-1] + 2 == Widths[0].
        self.d = Widths[-1];
        assert(Widths[0] == 2*self.d + 2);
        self.Widths = Widths;

        # Set up the network's layers.
        self.Layers     = torch.nn.ModuleList();
        for i in range(self.N_Layers):
            self.Layers.append(torch.nn.Linear(in_features = Widths[i], out_features = Widths[i + 1]));
            torch.nn.init.xavier_normal_(self.Layers[i].weight);

            torch.nn.init.zeros_(self.Layers[i].bias);
        
        # Finally, set the activation function.
        self.Activation = torch.nn.Softplus();



    def forward(self, x : torch.Tensor, y : torch.Tensor, tau : torch.Tensor, t : torch.Tensor) -> torch.Tensor:
        """
        This function passes x, y, and t through the neural network and returns F(x, y, tau, t) (see 
        class doc string).
        
        --------------------------------------------------------------------------------------------
        Arguments:

        x, y: 1D tensors representing the first two arguments of the right hand side of the above 
        DDE. These should have the same length, d. 
        
        tau: A single element tensor whose lone value represents the delay in the DDE (this only 
        matters if the delay appears explicitly in F).
        
        t: a single element tensor whose lone value represents the third argument to the DDE above.
        """

        # Checks.
        assert(len(x.shape) == 1);
        assert(len(y.shape) == 1);
        assert(x.numel()    == y.numel());
        assert(x.numel()    == self.d);
        assert(t.numel()    == 1);
        assert(tau.numel()  == 1);

        # Set up the input to the network.
        X : torch.Tensor = torch.concat([x.reshape(-1), y.reshape(-1), t.reshape(-1)], dim = 0);

        # Compute, return the output.  
        for i in range(self.N_Layers - 1):
            X = self.Activation(self.Layers[i](X));
        Output : torch.Tensor = self.Layers[-1](X);
        return Output;
