import  logging;

import  matplotlib      as  mpl;
import  colorsys;       
import  seaborn;



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
# Initialize logger
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