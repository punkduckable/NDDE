# -------------------------------------------------------------------------------------------------
# Setup 
# -------------------------------------------------------------------------------------------------

# Python Libraries
import  os;
from    typing              import      List, Dict, Tuple;

# External libraries. 
import  numpy;
import  matplotlib.pyplot       as      plt;
import  matplotlib              as      mpl;
import  yaml;
import  torch;
from    torch.utils.tensorboard import  SummaryWriter;
from    scipy                   import  interpolate;

# My code
from    Solver  import  RK2                 as DDE_Solver;
from    NDDE    import  NDDE;
from    Train   import  Train;
from    Utils   import  Initialize_Logger, Initialize_MPL, Add_Noise, Log_Dict;
from    Loss    import  L1_Cost, L2_Cost, Integral_Loss;

# Set up the TensorBoard SummaryWriter
Base_Dir    : str   = "./Logs";
Counter     : int   = 0;
Log_Dir     : str   = os.path.join(Base_Dir, "Version_" + str(Counter));
while(os.path.isdir(Log_Dir) == True):
    Counter += 1;
    Log_Dir  = os.path.join(Base_Dir, "Version_" + str(Counter));
TB_Writer = SummaryWriter(log_dir = Log_Dir);

# Set up the logger setup.
import  logging;
Initialize_Logger(level = logging.INFO);
LOGGER : logging.Logger = logging.getLogger(__name__);

# Set up cost functions.
l = L1_Cost(Weight = 1.0);
G = L1_Cost(Weight = 0.0);

# Set up plotting. 
Initialize_MPL();




# -------------------------------------------------------------------------------------------------
# Read in the settings.
# -------------------------------------------------------------------------------------------------

# Load, log the settings.
Config_File         = open("Experiment.conf", 'r');
Config      : dict  = yaml.safe_load(Config_File);
Log_Dict(LOGGER = LOGGER, D = Config);
Config_File.close();

# Next, let's make sure everything has the right type.
assert(isinstance(Config['Experiment'],             str));

assert(isinstance(Config['Noise Levels'],           list));
for Noise_Level in Config['Noise Levels']:
    assert(isinstance(Noise_Level, float));
    assert(Noise_Level > 0);

assert(isinstance(Config['Number of Experiments'],  int));
assert(Config['Number of Experiments'] >=    1);

assert(isinstance(Config['Number of Epochs'],       int));
assert(Config['Number of Epochs'] >=    0);

assert(isinstance(Config['Learning Rate'],          float));
assert(Config['Learning Rate'] > 0);

assert(isinstance(Config['Loss Threshold'],         float));



# -------------------------------------------------------------------------------------------------
# Set up the noise-free target trajectory
# -------------------------------------------------------------------------------------------------

# Log what we're doing.
LOGGER.info("Setting up a %s experiment!" % Config['Experiment']);

# Set up the Parameters
Parameter_Dict : Dict[str, float] = {};

# Now... set up the specific experiment.
# Note: If you want to add your own experiments, go for it! However, you do need to make sure 
# that each parameter has its own parameter.

# Exponential 
if(  Config['Experiment'] == "Exponential"):
    from    Model   import  Exponential         as F_Model;
    from    X0      import  Affine              as X0_Model;

    # Set up the parameters and tau value for the target trajectory.
    F_Target        = F_Model(theta_0 = -2.0, theta_1 = -2.0);
    a_Target        = torch.Tensor([1.5]);
    b_Target        = torch.Tensor([4.0]);
    X0_Target       = X0_Model(a = a_Target, b = b_Target);
    tau_Target      = torch.tensor(1.0);
    N_tau           = 10;
    T_Target        = torch.tensor(10.0);

# Logistic 
elif(Config['Experiment'] == "Logistic"):
    from    Model   import  Logistic            as F_Model;
    from    X0      import  Affine              as X0_Model;

    # Set up the parameters and tau value for the target trajectory.
    F_Target        = F_Model(theta_0 = 1.0, theta_1 = 1.0);
    a_Target        = torch.Tensor([0.5]);
    b_Target        = torch.Tensor([2]);
    X0_Target       = X0_Model(a = a_Target, b = b_Target);
    tau_Target      = torch.tensor(1.0);
    N_tau           = 10;
    T_Target        = torch.tensor(10.0);

# ESNO
elif(Config['Experiment'] == "ESNO"):
    from    Model   import  ENSO                as F_Model;
    from    X0      import  Periodic            as X0_Model;

    # Set up the parameters and tau value for the target trajectory.
    F_Target        = F_Model(a = 1.0, b = 1.0, c = 0.75);
    A_Target        = torch.Tensor([-0.25]);
    w_Target        = torch.Tensor([1.0]);
    b_Target        = torch.Tensor([1.5])
    X0_Target       = X0_Model(A = A_Target, w = w_Target, b = b_Target);
    tau_Target      = torch.tensor(5.0);
    N_tau           = 50;
    T_Target        = torch.tensor(10.0);

# Cheyne 
elif(Config['Experiment'] == "Cheyne"):
    from    Model   import  Cheyne              as F_Model;
    from    X0      import  Affine              as X0_Model;

    # Set up the parameters and tau value for the target trajectory.
    F_Target        = F_Model(p = 1.0, V0 = 7.0, a = 2.0, m = 8);
    a_Target        = torch.Tensor([-5.0]);
    b_Target        = torch.Tensor([2]);
    X0_Target       = X0_Model(a = a_Target, b = b_Target);
    tau_Target      = torch.tensor(0.25);
    N_tau           = 10;
    T_Target        = torch.tensor(3.0);

else:
    LOGGER.error("Invalid Experiment selection! Exiting");
    exit();

# Set up the true portion of the Parameter Dictionary
Parameter_Dict["True"] = {  "Tau"   : tau_Target.item() };

Parameter_Dict["True"]["Theta"] = {};
for key, value in F_Target._parameters.items():
     Parameter_Dict["True"]["Theta"][key] = value.item();

Parameter_Dict["True"]["Phi"] = {};
for key, value in X0_Target._parameters.items():
     Parameter_Dict["True"]["Phi"][key] = value.item();

# Confirm that we're done!
LOGGER.info("Done!");

# Now... make the true trajectory.
x_True, t_True    = DDE_Solver(F = F_Target, X0 = X0_Target, tau = tau_Target, T = T_Target, N_tau = N_tau);



# -------------------------------------------------------------------------------------------------
# Run the experiments!
# -------------------------------------------------------------------------------------------------

t_Target : torch.Tensor = t_True; 

for Noise_Level in Config['Noise Levels']:
    LOGGER.info("Running %d %s experiments at a noise level of %f" % (Config['Number of Experiments'], Config['Experiment'], Noise_Level));
    Parameter_Dict[Noise_Level] = {};

    for Experiment_index in range(Config['Number of Experiments']):
        LOGGER.info("Experiment %d/%d" % (Experiment_index + 1, Config['Number of Experiments']));

        # First, add noise.
        x_Target : torch.Tensor = Add_Noise(X = x_True, l = Noise_Level);

        # Next, set up the model.

        # Exponential
        if(Config['Experiment'] == "Exponential"):
            # Pick a starting position, tau, and x0
            tau     = (tau_Target*2.0).clone().detach().requires_grad_(True);
            a       = (a_Target*1.5).clone().detach().requires_grad_(True);
            b       = (b_Target*0.7).clone().detach().requires_grad_(True);
            T       = torch.clone(T_Target).requires_grad_(False);

            # Set up a NDDE object. We will try to train the enclosed model to match the one we used to generate the above plot.
            F_MODEL             = F_Model(theta_0 = -1.5, theta_1 = -2.5);
            X0_MODEL            = X0_Model(a, b);
            Param_List  : List  = [tau] + list(F_MODEL.parameters()) + list(X0_MODEL.parameters());


        # Logistic
        elif(Config['Experiment'] == "Logistic"):
            # Pick a starting position, tau, and x0
            tau     = (tau_Target*0.5).clone().detach().requires_grad_(True);
            a       = (a_Target*1.2).clone().detach().requires_grad_(True);
            b       = (b_Target*0.7).clone().detach().requires_grad_(True);
            T       = torch.clone(T_Target).requires_grad_(False);

            # Set up a NDDE object. We will try to train the enclosed model to match the one we used to generate the above plot.
            F_MODEL             = F_Model(theta_0 = 1.0, theta_1 = 1.0);
            X0_MODEL            = X0_Model(a, b);
            Param_List  : List  = [tau] + list(F_MODEL.parameters()) + list(X0_MODEL.parameters());


        # ESNO
        elif(Config['Experiment'] == "ESNO"):
            # Pick a starting position, tau, and x0
            tau     = (tau_Target*1.2).clone().detach().requires_grad_(True);
            A       = (A_Target*1.2).clone().detach().requires_grad_(True);
            w       = (w_Target*0.8).clone().detach().requires_grad_(True);
            b       = (b_Target*1.3).clone().detach().requires_grad_(True);
            T       = torch.clone(T_Target).requires_grad_(False);

            # Set up a NDDE object. We will try to train the enclosed model to match the one we used to generate the above plot.
            F_MODEL             = F_Model(a = 1.5, b = 0.8, c = 1.2);
            X0_MODEL            = X0_Model(A, w, b);
            Param_List  : List  = [tau] + list(F_MODEL.parameters()) + list(X0_MODEL.parameters());

        # Cheyne
        elif(Config['Experiment']   == "Cheyne"):
            # Pick a starting position, tau, and x0
            tau     = (tau_Target*2.0).clone().detach().requires_grad_(True);
            a       = (torch.zeros_like(a_Target)).clone().detach().requires_grad_(True);
            b       = (x_Target[0, :]).clone().detach().requires_grad_(True);
            T       = torch.clone(T_Target).requires_grad_(False);

            # Set up a NDDE object. We will try to train the enclosed model to match the one we used to generate the above plot.
            F_MODEL             = F_Model(p = 2.0, V0 = 12.0, a = 1.5, m = 8);
            X0_MODEL            = X0_Model(a, b);
            Param_List  : List  = [tau] + list(F_MODEL.parameters()) + list(X0_MODEL.parameters());

        else: 
            LOGGER.error("Invalid Experiment selection! Exiting");
            exit(); 

        # Set up the DDE module.
        DDE_Module  = NDDE(F = F_MODEL, X0 = X0_MODEL);

        # Now... set up training.
        Optimizer           = torch.optim.Adam(Param_List, lr = Config['Learning Rate']);
        Scheduler           = torch.optim.lr_scheduler.CosineAnnealingLR(
                                                    optimizer   = Optimizer, 
                                                    T_max       = Config['Number of Epochs'],
                                                    eta_min     = Config['Learning Rate']/10.);

        # Train!
        t_Predict, x_Predict  = Train(  DDE_Module          = DDE_Module, 
                                        tau                 = tau,
                                        N_tau               = N_tau, 
                                        T                   = T, 
                                        N_Epochs            = Config['Number of Epochs'], 
                                        x_Target            = x_Target, 
                                        t_Target            = t_Target, 
                                        l                   = l,
                                        G                   = G,
                                        Loss_Threshold      = Config['Loss Threshold'], 
                                        Optimizer           = Optimizer, 
                                        Scheduler           = Scheduler,
                                        Writer              = TB_Writer);

        # Record the final set of parameter for this experiment.
        Parameter_Dict[Noise_Level][Experiment_index] = {  "Tau"   : tau.item() };

        Parameter_Dict[Noise_Level][Experiment_index]["Theta"] = {};
        for key, value in F_MODEL._parameters.items():
            Parameter_Dict[Noise_Level][Experiment_index]["Theta"][key] = value.item();

        Parameter_Dict[Noise_Level][Experiment_index]["Phi"] = {};
        for key, value in X0_MODEL._parameters.items():
            Parameter_Dict[Noise_Level][Experiment_index]["Phi"][key] = value.item();



# -------------------------------------------------------------------------------------------------
# Report the results!
# -------------------------------------------------------------------------------------------------


# Report tau results.
print("---------------------------------------------");
print("                  Tau Table!                 ");
print("---------------------------------------------");
print("True          %f" % Parameter_Dict["True"]["Tau"]);
for Noise_Level in Config['Noise Levels']:
    print("---------------------------------------------");
    print("              Noise level %5.2f              " % Noise_Level);
    print("---------------------------------------------");
    for Experiment_index in range(Config["Number of Experiments"]):
        print("%3d           %f" % (Experiment_index, Parameter_Dict[Noise_Level][Experiment_index]["Tau"]));

# Report Theta result
print("---------------------------------------------");
print("                 Theta Table!                ");
print("---------------------------------------------");
Line : str = "     ";
for (key, item) in F_Target._parameters.items():
    Line += " %7s " % key;
print(Line);

Line : str = "True ";
for (key, item) in F_Target._parameters.items():
    Line += " %7.3f " % Parameter_Dict["True"]["Theta"][key];
print(Line);

for Noise_Level in Config['Noise Levels']:
    print("---------------------------------------------");
    print("              Noise level %5.2f              " % Noise_Level);
    print("---------------------------------------------");
    for Experiment_index in range(Config["Number of Experiments"]):
        Line : str = "%4d " % Experiment_index;
        for (key, item) in F_Target._parameters.items():
            Line += " %7.3f " % Parameter_Dict[Noise_Level][Experiment_index]["Theta"][key];
        print(Line);
    

# Report Phi result
print("---------------------------------------------");
print("                  Phi Table!                 ");
print("---------------------------------------------");
Line : str = "     ";
for (key, item) in X0_Target._parameters.items():
    Line += " %7s " % key;
print(Line);

Line : str = "True ";
for (key, item) in X0_Target._parameters.items():
    Line += " %7.3f " % Parameter_Dict["True"]["Phi"][key];
print(Line);

for Noise_Level in Config['Noise Levels']:
    print("---------------------------------------------");
    print("              Noise level %5.2f              " % Noise_Level);
    print("---------------------------------------------");
    for Experiment_index in range(Config["Number of Experiments"]):
        Line : str = "%4d " % Experiment_index;
        for (key, item) in X0_Target._parameters.items():
            Line += " %7.3f " % Parameter_Dict[Noise_Level][Experiment_index]["Phi"][key];
        print(Line);