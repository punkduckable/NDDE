import  logging;
from    typing      import Dict, Callable, Tuple;

import  torch;
import  numpy;
from    scipy       import  interpolate;
from    Loss        import  SSE_Loss, Integral_Loss;

# Minimum allowed value of tau
Tau_Threshold   : float = 0.01;

# Set up the logger
LOGGER : logging.Logger = logging.getLogger(__name__);



def Train(  DDE_Module      : torch.nn.Module, 
            tau             : torch.Tensor, 
            T               : torch.Tensor,
            N_Epochs        : int, 
            x_Target        : torch.Tensor, 
            t_Target        : torch.Tensor, 
            l               : torch.nn.Module,
            G               : torch.nn.Module,
            Loss_Threshold  : float, 
            Optimizer       : torch.optim.Optimizer,
            Scheduler                   = None,
            Writer                      = None) -> Tuple[numpy.ndarray, numpy.ndarray]:
    """
    This function implements the training loop for a simple NDDE object.

    TO DO
    """

    # ---------------------------------------------------------------------------------------------
    # Checks!
    assert(tau.numel()      == 1);
    assert(T.numel()        == 1);
    assert(isinstance(N_Epochs, int));
    assert(N_Epochs         >= 0);
    assert(tau.item()       >  0);

    LOGGER.debug("Checks passed");


    # ---------------------------------------------------------------------------------------------
    # Setup 

    # Set up buffers to track history of loss, and tau.
    History_Dict : Dict[str, torch.Tensor] = {};
    History_Dict["Loss"]    = torch.zeros(N_Epochs);
    History_Dict["tau"]     = torch.zeros(N_Epochs);

    # Set up an interpolation for the target trajectory. We will need to evaluate this 
    # wherever we evaluate the predicted trajectory
    x_Target_Interpolated                         = interpolate.interp1d(t_Target.detach().numpy(), x_Target.detach().numpy());


    # ---------------------------------------------------------------------------------------------
    # Run the epochs!

    for epoch in range(1, N_Epochs + 1):   
        LOGGER.debug("Starting epoch #%d" % epoch); 

        # Set up variables that we want to return.
        t_Predict_np            : numpy.ndarray = None;
        Predicted_Trajectory    : torch.Tensor  = None;

        def Closure() -> torch.Tensor:
            # -----------------------------------------------------------------------------------------
            # Set up 
            
            # Zero the gradients. 
            Optimizer.zero_grad();

            # Overwrite the versions of these variables defined above.
            nonlocal t_Predict_np;
            nonlocal Predicted_Trajectory;


            # -----------------------------------------------------------------------------------------
            # Compute the loss.

            # find the predicted trajectories with current tau, parameter values.
            Predicted_Trajectory                    = DDE_Module(tau, T, l, G, x_Target_Interpolated);
            xT_Predict              : torch.Tensor  = Predicted_Trajectory[-1];

            # find the time steps for the output trajectory
            N_Steps : int = Predicted_Trajectory.shape[1];

            # interpolate the target solution at the new time steps. Note that we need to do 
            # this every epoch because tau changes each epoch, and tau controls the step size.
            t_Predict_np                            = numpy.linspace(start = 0, stop = T.item(), num = N_Steps);
            Target_Trajectory       : torch.Tensor  = torch.from_numpy(x_Target_Interpolated(t_Predict_np));

            xT_Target               : torch.Tensor  = Target_Trajectory[-1];

            # Compute the loss!
            Loss_Terminal           : torch.Tensor  = G(xT_Predict, xT_Target);
            Loss_Running            : torch.Tensor  = Integral_Loss(Predicted_Trajectory, Target_Trajectory, torch.from_numpy(t_Predict_np), l);
            Loss                    : torch.Tensor  = Loss_Terminal + Loss_Running;
            LOGGER.debug("Loss = %f" % Loss.item());
            

            # -----------------------------------------------------------------------------------------
            # Run back-propagation, log, return!

            # Backprop!
            Loss.backward();

            # Log!
            if(Writer is not None):
                Writer.add_scalar(r"$\tau$",                tau.item(),                                             epoch);
                Writer.add_scalar(r"$\nabla \tau$",         tau.grad.item(),                                        epoch);
                Writer.add_scalar(r"Loss_{Total}",          Loss.item(),                                            epoch);
                Writer.add_scalar(r"Loss_{Terminal}",       Loss_Terminal.item(),                                   epoch);
                Writer.add_scalar(r"Loss_{Running}",        Loss_Running.item(),                                    epoch);
        
            # All done
            return Loss;


        # -----------------------------------------------------------------------------------------
        # Update the parameters.

        # Otherwise, run one step of the optimizer!
        Loss : torch.Tensor =  Optimizer.step(closure = Closure);

        # Run the learning rate scheduler, if it exists
        if(Scheduler is not None): Scheduler.step();


        # -----------------------------------------------------------------------------------------
        # Check for stopping condition, invalid tau

        # Check if loss is low enough to stop
        if Loss < Loss_Threshold:
            print("converged after %d epochs" % epoch);
            break;

        # Check if tau is below the threshold. If so, reset it.
        if(tau.item() < Tau_Threshold):
            LOGGER.warning("Tau dropped below allowed minimum (%f). Current value is %f." % 
                            (Tau_Threshold, tau.item()));
            LOGGER.warning("Resetting tau to %f" % Tau_Threshold);

            # Reset tau and its gradient.
            tau         = torch.tensor(Tau_Threshold, requires_grad = True);
            tau.grad    = torch.zeros_like(tau);
    

        # -----------------------------------------------------------------------------------------
        # Report!

        if epoch % 10 == 0:
            print(  "%4d: "                         % epoch,
                    " Loss = %7.5f"                 % Loss.item(), 
                    " | tau = %7.5f"                % tau.item(), 
                    " | grad tau = %9.5f"           % tau.grad.item());
            #plt.plot(Predicted_Trajectory[0].detach().numpy());

        # Save the data for printing later
        History_Dict["Loss"][epoch - 1]     = Loss.item();
        History_Dict["tau"][epoch - 1]      = tau.item();

    # Report final tau, parameter values.
    LOGGER.debug("Final values:");

    # All done... return!
    return t_Predict_np, Predicted_Trajectory.detach();