import torch; 



class MODEL(torch.nn.Module):
    """ 
    Objects of the MODEL class are designed to model the right-hand side of a DDE. Consider the 
    following DDE:
        x'(t) = F(x(t), x(t - \tau), t)     t \in [0, T]
        x(t)  = x_0                         t \in [-\tau, 0]
    A MODEL object is supposed to act like the function F in the expression above. In other words, 
    it defines a function which accepts three arguments: x, y, and t. If x = x(t), y = x(t - \tau),
    then the returned value  can be interpreted as the right hand side of a DDE.
    """

    def __init__(self, c_0 : float = 2.0, c_1 : float = 2.0):
        """ 
        Currently, this MODEL class is set up to act as a logistic map:
            F(x, y, t) = c_0*x*(1 - c_1*y)
        (there is no explicit dependence on t). Thus, the arguments c_0 and c_1 define the function 
        implied by this MODEL object.
        """
        
        # Call the super class initializer. 
        super(MODEL, self).__init__();

        # Set model parameters.
        self.Params = torch.nn.parameter.Parameter(torch.tensor([c_0, c_1], dtype = torch.float32, requires_grad = True));



    def forward(self, x : torch.Tensor, y : torch.Tensor, t : torch.Tensor) -> torch.Tensor:
        """
        A MODEL object defines the right hand side of a DDE (see class doc-string above). Thus, the 
        forward map defines that function. We expect x, y, and t to be single element tensors. This 
        function then returns
            F(x, y, t) = c_0*x*(1 - c_1*y)
        
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