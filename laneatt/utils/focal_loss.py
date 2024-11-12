from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

def one_hot(labels: torch.Tensor,
            num_classes: int,
            device: Optional[torch.device] = None,
            dtype: Optional[torch.dtype] = None,
            eps: Optional[float] = 1e-6) -> torch.Tensor:
    """
        Convert an integer label tensor to a one hot tensor.

        Args:
            labels (torch.Tensor) : tensor with labels of shape (n_positve+n_negative) indicating the class with an index [0, 1].
            num_classes (int): number of classes in labels.
            device (Optional[torch.device]): the desired device of the returned tensor.
            dtype (Optional[torch.dtype]): the desired data type of the returned tensor.
            eps (Optional[float]): small value to avoid division by zero.

        Returns:
            torch.Tensor: the labels in one hot tensor [1,  -> [[0, 1]
                                                        1,       [0, 1]
                                                        1,       [0, 1]
                                                        0,       [1, 0]
                                                        0,       [1, 0]
                                                        1,]      [0, 1]]
    """
    # Data validation
    if not torch.is_tensor(labels):
        raise TypeError("Input labels type is not a torch.Tensor. Got {}".format(type(labels)))
    if not labels.dtype == torch.int64:
        raise ValueError("labels must be of the same dtype torch.int64. Got: {}".format(labels.dtype))
    if num_classes < 1:
        raise ValueError("The number of classes must be bigger than one." " Got: {}".format(num_classes))
    
    # Compute the one hot tensor
    shape = labels.shape
    one_hot = torch.zeros(shape[0], num_classes, *shape[1:], device=device, dtype=dtype)
    return one_hot.scatter_(1, labels.unsqueeze(1), 1.0) + eps

def focal_loss(input: torch.Tensor,
               target: torch.Tensor,
               alpha: float,
               gamma: float = 2.0,
               eps: float = 1e-8) -> torch.Tensor:
    """
        Function that computes Focal loss.

        Args:
            input (torch.Tensor): the input tensor with the logits.
            target (torch.Tensor): the target tensor with the labels indices.
            alpha (float): Weighting factor in range (0,1) to balance positive vs negative samples.
            gamma (float): Focusing parameter for modulating factor (1 - p_t)^gamma.
            eps (float): Scalar to enforce numerical stability.

        Returns:
            torch.Tensor: the computed loss for each proposal.
    """
    if not torch.is_tensor(input):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(type(input)))

    if not len(input.shape) >= 2:
        raise ValueError("Invalid input shape, we expect BxCx*. Got: {}".format(input.shape))

    if input.size(0) != target.size(0):
        raise ValueError('Expected input batch_size ({}) to match target batch_size ({}).'.format(
            input.size(0), target.size(0)))

    # Get the number of proposals
    n = input.size(0)
    out_size = (n, ) + input.size()[2:]
    if target.size()[1:] != input.size()[2:]:
        raise ValueError('Expected target size {}, got {}'.format(out_size, target.size()))

    if not input.device == target.device:
        raise ValueError("input and target must be in the same device. Got: {} and {}".format(
            input.device, target.device))

    # Compute softmax over the classes axis
    input_soft = F.softmax(input, dim=1) + eps

    # Create the labels one hot tensor
    target_one_hot = one_hot(target, num_classes=input.shape[1], device=input.device, dtype=input.dtype)

    # Compute the actual focal loss
    weight = torch.pow(-input_soft + 1., gamma)
    focal = -alpha * weight * torch.log(input_soft)

    # Compute the loss summing over all classes since the one hot tensor will discard the negative class
    loss = torch.sum(target_one_hot * focal, dim=1)
    return loss

class FocalLoss(nn.Module):
    def __init__(self, alpha: float, gamma: float = 2.0) -> None:
        """
            Focal loss function initialization.

            Args:
                alpha (float): Weighting factor in range (0,1) to balance positive vs negative samples.
                gamma (float): Focusing parameter for modulating factor (1 - p_t)^gamma.
        """
        super(FocalLoss, self).__init__()
        self.alpha: float = alpha
        self.gamma: float = gamma
        self.eps: float = 1e-6

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
            Forward pass of the focal loss.

            Args:
                input (torch.Tensor): the input tensor with the logits.
                target (torch.Tensor): the target tensor with the labels indices.

            Returns:
                torch.Tensor: the computed loss for each proposal.
        """
        return focal_loss(input, target, self.alpha, self.gamma, self.eps)