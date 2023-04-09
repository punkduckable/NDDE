import  torch; 
from    typing      import Dict, Tuple;

from    NDDE        import NDDE_1D;
from    MODEL       import MODEL;
from    Interpolate import Interpolate_Trajectory;
from    Loss        import MSE_Loss     as Loss_Fn;



def Train_2_Param(  
            N_Epochs            : int, 
            DDE_Module          : NDDE_1D, 
            tau                 : torch.Tensor,
            x_0                 : torch.Tensor,
            T                   : torch.Tensor, 
            x_trajectory_True   : torch.Tensor, 
            t_trajectory_True   : torch.Tensor, 
            Optimizer           : torch.optim.Optimizer) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    """
    This function trains tau and the Model's parameters such that the trajectory with that tau and
    those parameter values matches the trajectory that created x_trajectory_True. 

    Note: This function only works if DDE_Module has two parameters. I need to generalize this in 
    the future.

    -----------------------------------------------------------------------------------------------
    Arguments: 

    N_Epochs: The number of epochs we train for. Must be a positive integer.

    DDE_Module: A NDDE_1D object which houses the model we want to train.

    tau: A single element tensor whose lone element specifies our current best guess for tau.
    
    x_0: a 1D tensor holding the initial coordinates of the DDE system.
    
    T: a single element tensor whose lone element specifies the final in the DDE solve.

    x_trajectory_True: a 1D tensor whose ith column holds the start of the true solution at the ith 
    time step (ith element of t_trajectory_True)
    """


    # Run checks.
    assert(len(x_0.shape)               == 1);
    assert(tau.numel()                  == 1);
    assert(T.numel()                    == 1);
    assert(len(x_trajectory_True.shape) == 2);
    assert(len(t_trajectory_True.shape) == 1);
    assert(x_trajectory_True.shape[0]   == x_0.shape[0]);
    assert(t_trajectory_True.shape[0]   == x_trajectory_True.shape[1]);
    assert(N_Epochs                     >  0);

    # Set up buffers to track history of loss, tau, and parameters.
    History_Dict : Dict[str, torch.Tensor] = {};
    History_Dict["Loss"]    = torch.zeros(N_Epochs);
    History_Dict["tau"]     = torch.zeros(N_Epochs);

    # First, extract the model. 
    Model : MODEL = DDE_Module.Model;

    # Now, run the epochs!
    for epoch in range(N_Epochs):    
        # find the predicted trajectories with current tau, parameter values.
        Predicted_Trajectory : torch.Tensor = DDE_Module(x_0, tau, T);

        if torch.any(torch.isnan(Predicted_Trajectory)) == False:
            # find the time steps for the output trajectory
            N : int = Predicted_Trajectory.shape[1];

            # interpolate the data at the new time steps. Note that we need to do this 
            # every epoch because tau changes each epoch, and tau controls the step size.
            Target_Trajectory : torch.Tensor = Interpolate_Trajectory(
                                                x_trajectory    = x_trajectory_True, 
                                                t_trajectory    = t_trajectory_True,
                                                N_Interp        = N);

            # Compute the loss!
            Loss : torch.Tensor = Loss_Fn(Predicted_Trajectory, Target_Trajectory, tau);

            # Check if loss is low enough to stop
            if Loss < 0.01:
                print("converged after %d epochs" % epoch);
                break;

            # Otherwise, run one step of the optimizer!
            Optimizer.zero_grad();
            Loss.backward();
            Optimizer.step();

            # Report loss and stuff.
            if epoch % 10 == 0:
                print(  "%4d: "      % epoch,
                        " Loss = %7.5f" % Loss.item(), 
                        " | tau = %7.5f" % tau.item(), 
                        " | grad tau = %9.5f" % tau.grad.item(),
                        " | Params = %7.5f, %7.5f" % (Model.Params[0], Model.Params[1]), 
                        " | grad Params = %9.5f, %9.5f" % (Model.Params.grad[0], Model.Params.grad[1]));
                #plt.plot(Predicted_Trajectory[0].detach().numpy());

            # save the data for printing later
            History_Dict["Loss"][epoch]         = Loss.item();
            History_Dict["tau"][epoch]          = tau.item();

        else:    
            print("Something went wrong :(");
            break;

    # Report final tau, parameter values.
    print("Final values:");
    print("tau = %7.5f, c_0 = %7.5f, c_1 = %7.5f" % (tau.item(), Model.Params[0], Model.Params[1]));

    # All done!
    return History_Dict, Predicted_Trajectory;