import torch; 

# Logger setup 
import logging;
LOGGER : logging.Logger = logging.getLogger(__name__);



class Constant(torch.nn.Module):
    """
    This class implements a constant IC function. 
    """
    def __init__(self, x0 : torch.Tensor) -> None:
        """
        -------------------------------------------------------------------------------------------
        Arguments:
        
        x0 : This should be a single element tensor whose lone value holds the constant we want to 
        set the IC to.
        """

        # Run the super class initializer. 
        super(Constant, self).__init__();

        # Run checks
        assert(len(x0.shape) == 1);

        # Store the constant x0 value as a parameter object.
        self.x0 = torch.nn.Parameter(x0, requires_grad = True);


    def forward(self, t : torch.Tensor) -> torch.Tensor:
        """
        -------------------------------------------------------------------------------------------
        Arguments:
        
        t : This should be a 1D torch.Tensor whose i'th value holds the i'th t value.

        
        -------------------------------------------------------------------------------------------
        Returns: 

        A 1D torch.Tensor object whose i'th value holds x0.
        """

        # The IC is ALWAYS x0... it's a constant!
        return self.x0.repeat(list(t.shape) + [1]);



class Affine(torch.nn.Module):
    """
    This class implements a simple affine IC:
        X0(t) = a*t + b
    """
    def __init__(self, a : torch.Tensor, b : torch.Tensor) -> None:
        """
        -------------------------------------------------------------------------------------------
        Arguments:
        
        x0 : This should be a single element tensor whose lone value holds the constant we want to 
        set the IC to.
        """

        # Run the super class initializer. 
        super(Affine, self).__init__();

        # Run checks
        assert(len(a.shape) == 1);
        assert(len(b.shape) == 1);

        # Store the constants a, b as parameters.
        self.a = torch.nn.Parameter(a.reshape(1, -1), requires_grad = True);
        self.b = torch.nn.Parameter(b.reshape(1, -1), requires_grad = True);



    def forward(self, t : torch.Tensor) -> torch.Tensor:
        """
        -------------------------------------------------------------------------------------------
        Arguments:
        
        t : This should be a 1D torch.Tensor whose i'th value holds the i'th t value.

        
        -------------------------------------------------------------------------------------------
        Returns: 

        A 1D torch.Tensor object whose i'th value holds x0.
        """

        # Reshape t.
        t = t.reshape(-1, 1);

        # The IC is ALWAYS x0... it's a constant!
        return (self.a)*t + self.b;

