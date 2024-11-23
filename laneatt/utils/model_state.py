import matplotlib.pyplot as plt
import os
import re
import torch
import pickle

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
    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)
    
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
    if not os.path.exists(checkpoints_dir): raise FileNotFoundError('Checkpoints directory not found.')
    
    # Generate the pattern to match the checkpoint files and a list of all the checkpoint files
    pattern = re.compile('laneatt_(\\d+).pt')
    checkpoints = [ckpt for ckpt in os.listdir(checkpoints_dir) if re.match(pattern, ckpt) is not None]
    if len(checkpoints) == 0: raise FileNotFoundError('No checkpoint files found.')

    # Get last checkpoint epoch
    latest_checkpoint_path = sorted(checkpoints, reverse=True, key=lambda name : int(name.split('_')[1].rstrip('.pt')))[0]
    epoch = latest_checkpoint_path.split('_')[1].rstrip('.pt')

    return latest_checkpoint_path, int(epoch)

def save_data(data, data_dir, data_name):
    """
        Save the data to the data directory.

        Args:
            data: The data to be saved.
            data_dir: The directory to save the data.
            data_name: The name of the data file.
    """
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    if not os.path.exists(os.path.join(data_dir, data_name)):
        with open(os.path.join(data_dir, data_name), 'wb') as f:
            pickle.dump([data], f)
    else:
        with open(os.path.join(data_dir, data_name), 'rb') as f:
            read_data = pickle.load(f)
        read_data.append(data)
        with open(os.path.join(data_dir, data_name), 'wb') as f:
            pickle.dump(read_data, f)

def remove_data(data_dir):
    """
        Remove the data from the data directory.

        Args:
            data_dir: The directory to remove the data.
    """
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    elements = os.listdir(data_dir)
    for element in elements:
        os.remove(os.path.join(data_dir, element) if os.path.isfile(os.path.join(data_dir, element)) else None)

def plot_from_data(data_dir):
    """
        Plot the data from the data directory.

        Args:
            data_dir: The directory to plot the data.
    """
    if not os.path.exists(data_dir): raise FileNotFoundError('Data directory not found.')
    
    data_files = [f for f in os.listdir(data_dir) if f.endswith('.pkl')]

    train_loss, train_cls_loss, train_reg_loss = [], [], []
    eval_loss, eval_cls_loss, eval_reg_loss, precision, recall, f1, accuracy = [], [], [], [], [], [], []
    for data_file in data_files:
        data_type = data_file.split('_')[0]

        with open(os.path.join(data_dir, data_file), 'rb') as f:
            data = pickle.load(f)

        if data_type == 'train':
            for entry in data:
                train_loss.append(entry['loss'])
                train_cls_loss.append(entry['cls_loss'])
                train_reg_loss.append(entry['reg_loss'])
        elif data_type == 'eval':
            for entry in data:
                eval_loss.append(entry['loss'])
                eval_cls_loss.append(entry['cls_loss'])
                eval_reg_loss.append(entry['reg_loss'])
                precision.append(entry['precision'])
                recall.append(entry['recall'])
                f1.append(entry['f1_score'])
                accuracy.append(entry['accuracy'])
        
    # Plot the data
    ax, fig = plt.subplots(2, 5, figsize=(20, 10))
    fig[0, 0].plot(train_loss)
    fig[0, 0].set_title('Train Loss')
    fig[0, 1].plot(train_cls_loss)
    fig[0, 1].set_title('Train Classification Loss')
    fig[0, 2].plot(train_reg_loss)
    fig[0, 2].set_title('Train Regression Loss')
    fig[1, 0].plot(eval_loss)
    fig[1, 0].set_title('Eval Loss')
    fig[1, 1].plot(eval_cls_loss)
    fig[1, 1].set_title('Eval Classification Loss')
    fig[1, 2].plot(eval_reg_loss)
    fig[1, 2].set_title('Eval Regression Loss')
    fig[0, 3].plot(precision)
    fig[0, 3].set_title('Precision')
    fig[0, 4].plot(recall)
    fig[0, 4].set_title('Recall')
    fig[1, 3].plot(f1)
    fig[1, 3].set_title('F1 Score')
    fig[1, 4].plot(accuracy)
    fig[1, 4].set_title('Accuracy')

    plt.savefig(os.path.join(data_dir, f'metrics.png'))