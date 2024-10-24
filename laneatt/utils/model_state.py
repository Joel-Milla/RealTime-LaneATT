import os
import re
import torch

def load_last_train_state(model, optimizer, scheduler, checkpoints_dir):
    """
        Load the last training state from the checkpoint files.

        Args:
            model: The model to be loaded.
            optimizer: The optimizer to be loaded.
            scheduler: The scheduler to be loaded.
            config: The configuration of the training.
        
        Returns:
            epoch: The epoch of the last checkpoint.
            model: The model loaded from the last checkpoint.
            optimizer: The optimizer loaded from the last checkpoint.
            scheduler: The scheduler loaded from the last checkpoint.
    """
    
    train_state_path, epoch = get_last_checkpoint(checkpoints_dir)
    train_state = torch.load(os.path.join(checkpoints_dir, train_state_path), weights_only=True)
    model.load_state_dict(train_state['model'])
    optimizer.load_state_dict(train_state['optimizer'])
    scheduler.load_state_dict(train_state['scheduler'])

    return epoch, model, optimizer, scheduler

def save_train_state(epoch, model, optimizer, scheduler, checkpoints_dir):
    """
        Save the training state to the checkpoint files.

        Args:
            epoch: The epoch of the current training state.
            model: The model to be saved.
            optimizer: The optimizer to be saved.
            scheduler: The scheduler to be saved.
    """
    
    train_state = {
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict()
    }
    torch.save(train_state, os.path.join(checkpoints_dir, f'laneatt_{epoch}.pt'))

def get_last_checkpoint(checkpoints_dir):
    """
        Get the epoch of the last checkpoint.

        Returns:
            The epoch of the last checkpoint.
    """
    
    # Generate the pattern to match the checkpoint files and a list of all the checkpoint files
    pattern = re.compile('laneatt_(\\d+).pt')
    checkpoints = [ckpt for ckpt in os.listdir(checkpoints_dir) if re.match(pattern, ckpt) is not None]
    if len(checkpoints) == 0: raise FileNotFoundError('No checkpoint files found.')

    # Get last checkpoint epoch
    latest_checkpoint_path = sorted(checkpoints, reverse=True, key=lambda name : int(name.split('_')[1].rstrip('.pt')))[0]
    epoch = latest_checkpoint_path.split('_')[1].rstrip('.pt')

    return latest_checkpoint_path, int(epoch)