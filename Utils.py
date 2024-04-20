import  logging;

import  matplotlib      as  mpl;
import  colorsys;       
import  seaborn;
import  torch;



# -------------------------------------------------------------------------------------------------
# Initialize logger
# -------------------------------------------------------------------------------------------------

def Initialize_Logger(level : int) -> None:
    """
    This function initializes and configures the logger.

    
    # ---------------------------------------------------------------------------------------------
    # Arguments
    # ---------------------------------------------------------------------------------------------

    level: This should be one of logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, or 
    logging.CRITICAL.


    # ---------------------------------------------------------------------------------------------
    # Returns 
    # ---------------------------------------------------------------------------------------------

    Nothing!
    """

    # Initialize the logger, set the level.
    logger = logging.getLogger();
    logger.setLevel(level);

    # Set up a handler to pass logged info to the console
    sh = logging.StreamHandler();
    sh.setLevel(level);
    
    # Setup a formatter for the handler. 
    LOG_FMT : str = '%(asctime)s | %(filename)s:%(funcName)s:%(lineno)s | %(levelname)s - %(message)s';
    sh.setFormatter(logging.Formatter(LOG_FMT));
    logger.addHandler(sh);



# -------------------------------------------------------------------------------------------------
# Initialize MatPlotLib
# -------------------------------------------------------------------------------------------------

def Initialize_MPL() -> None:
    """
    This function initialize matplotlib's parameter so that the plots it generates are less ugly.
    This function has no arguments or return variables.
    """

    # Now... let's set up plot formatting.
    def scale_lightness(rgb, scale_l):
        # convert rgb to hls
        h, l, s = colorsys.rgb_to_hls(*rgb)

        # manipulate h, l, s values and return as rgb
        return colorsys.hls_to_rgb(h, min(1, l * scale_l), s = s)

    seaborn.set_context(context     = "paper");
    seaborn.set_style(  style       = "darkgrid");
    mpl.rcParams["lines.linewidth"] = 2;
    mpl.rcParams["axes.linewidth"]  = 1.5;
    mpl.rcParams["axes.edgecolor"]  = "black";
    mpl.rcParams["grid.color"]      = "gray";
    mpl.rcParams["grid.linestyle"]  = "dotted";
    mpl.rcParams["grid.linewidth"]  = .67;
    mpl.rcParams["xtick.labelsize"] = 10;
    mpl.rcParams["ytick.labelsize"] = 10;
    mpl.rcParams["axes.labelsize"]  = 11;
    mpl.rcParams["axes.titlesize"]  = 12;
    mpl.rcParams["axes.facecolor"]  = scale_lightness(mpl.colors.ColorConverter.to_rgb("lightgrey"), 1.15);



# -------------------------------------------------------------------------------------------------
# Add Noise
# -------------------------------------------------------------------------------------------------

def Add_Noise(X : torch.Tensor, l : float) -> torch.Tensor:
    """
    This function adds Gaussian white noise to the dataset, X. How do we do this? We assume that X
    is a N x d tensor. For each j \in {0, 1, ... , d - 1}, we find the STD of X[:, j], which we 
    will denote by sigma_i. For each i \in {0, 1, ... , N - 1}, we sample a Gaussian random 
    variable with mean 0 and STD l*sigma_i, then add that sample to X[i, j]. We do this for each 
    i, j and then return the result. 

    
    -----------------------------------------------------------------------------------------------
    Arguments:

    X: A 2D, N x d tensor. We add Gaussian noise to each column of X such that the noise in 
    column j has a STD of l times the STD of the values in X[:, j].

    l: The noise level. This should be a non-negative float.
    """

    # Checks
    assert(isinstance(l, float));        
    assert(l >= 0);
    assert(len(X.shape) == 2); 

    # Edge cases
    if(l == 0):
        return X;

    # Recover information about X.
    N : int = X.shape[0];
    d : int = X.shape[1];

    # Now... add noise to X, column by column. To do this, we first make an tensor of "per-element"
    # standard deviations. This tensor has the same shape as X. To make this, we first compute the
    # STD of each column of x, but make sure to keep the extra dimension. This gives us a "sigma" 
    # tensor of shape 1 x d. We then expand this tensor along the 0 axis to have shape N x d. This 
    # technically doesn't create any new memory, it just makes a tensor-like thing, S, such that 
    # S[i, j] returns sigma[j] for each i, j. Then, for each i,j, we use torch's normal function to 
    # sample a normal distribution with mean X[i, j] and standard deviation sigma[j]. We store 
    # these samples in a new tensor, Y, which is what we return.
    sigma   : torch.Tensor  = l*torch.std(X, dim = 0, keepdim = True);
    S       : torch.Tensor  = sigma.expand(N, -1);
    Y       : torch.Tensor  = torch.normal(mean = X, std = S);

    return Y;



# -------------------------------------------------------------------------------------------------
# Log a dictionary
# -------------------------------------------------------------------------------------------------

def Log_Dict(LOGGER : logging.Logger, D : dict, level : int = logging.DEBUG, indent : int = 0) -> None:
    indent_str : str = '   '*indent;

    # Determine which level we're using to report the dictionary. Can either be debug or info.
    if(  level == logging.DEBUG):
        Report = LOGGER.debug;
    elif(level == logging.INFO):
        Report = LOGGER.info;
    else:
        LOGGER.warning("Invalid dictionary log level. Valid options are logging.DEBUG = %d and logging.INFO = %d. Got %d" % (logging.DEBUG, logging.INFO, level));
        LOGGER.warning("Returning without reporting dictionary...");
        return; 

    # Report the dictionary
    for k,v in D.items():
        if(isinstance(v, dict)):
            Report("%s[%s] ==>" % (indent_str, str(k)));
            Log_Dict(LOGGER = LOGGER, D = v, level = level, indent = indent + 1);   
        else:
            Report("%s[%s] ==> [%s]" % (indent_str, str(k), str(v)));