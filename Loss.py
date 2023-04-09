import torch;



def MSE_Loss(predictions : torch.Tensor, targets : torch.Tensor, tau : torch.Tensor, l : float = 0.0) -> torch.Tensor:
    return torch.mean(torch.square(predictions - targets)) + l*tau**2;