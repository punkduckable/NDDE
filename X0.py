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
        super().__init__();

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
        
        a, b : These should be 1D tensors whose holding the constants in the function t -> a*t + b.
        """

        # Run the super class initializer. 
        super().__init__();

        # Run checks
        assert(len(a.shape) == 1);
        assert(len(b.shape) == 1);
        assert(a.shape[0]   == b.shape[0]);

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

        # Compute the IC!
        return (self.a)*t + self.b;



class Periodic(torch.nn.Module):
    """
    This class implements a simple periodic IC:
        X0(t) = A*cos(w*t)
    """
    def __init__(self, A : torch.Tensor, w : torch.Tensor) -> None:
        """
        -------------------------------------------------------------------------------------------
        Arguments:
        
        A, w: These should be 1D tensors whose k'th components define the k'th component of the 
        initial condition: X0_k(t) = A_k * cos(w_k * t).
        """

        # Run the super class initializer. 
        super().__init__();

        # Run checks
        assert(len(A.shape) == 1);
        assert(len(w.shape) == 1);
        assert(A.shape[0]   == w.shape[0]);

        # Store the constants A, w as parameters.
        self.A = torch.nn.Parameter(A.reshape(1, -1), requires_grad = True);
        self.w = torch.nn.Parameter(w.reshape(1, -1), requires_grad = True);



    def forward(self, t : torch.Tensor) -> torch.Tensor:
        """
        -------------------------------------------------------------------------------------------
        Arguments:
        
        t : This should be a 1D torch.Tensor whose i'th value holds the i'th t value.

        
        -------------------------------------------------------------------------------------------
        Returns: 

        A 1D torch.Tensor object whose i'th value holds X0(t[i]).
        """

        # Reshape t.
        t = t.reshape(-1, 1);

        # Compute the IC!
        return torch.mul(self.A, torch.cos(torch.mul(t, self.w)));
