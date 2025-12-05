"""Visualization functions for time series"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_time_series(data, title="Time Series", save_path=None):
    """Plot multiple time series"""
    plt.figure(figsize=(15, 5))
    
    if len(data.shape) == 3:  # (n_samples, seq_len, n_features)
        for i in range(min(10, data.shape[0])):
            plt.plot(data[i, :, 0], alpha=0.5)
    elif len(data.shape) == 2:  # (seq_len, n_features)
        plt.plot(data[:, 0])
    
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def plot_comparison(real_data, generated_data, save_path=None):
    """Compare real and generated time series"""
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    
    # Plot real data
    for i in range(5):
        axes[0, i].plot(real_data[i, :, 0])
        axes[0, i].set_title(f'Real {i+1}')
        axes[0, i].grid(True, alpha=0.3)
    
    # Plot generated data
    for i in range(5):
        axes[1, i].plot(generated_data[i, :, 0])
        axes[1, i].set_title(f'Generated {i+1}')
        axes[1, i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def plot_distribution_comparison(real_data, generated_data, save_path=None):
    """Compare distributions of real and generated data"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Mean distribution
    real_means = real_data.mean(axis=1).flatten()
    gen_means = generated_data.mean(axis=1).flatten()
    axes[0].hist(real_means, bins=30, alpha=0.5, label='Real')
    axes[0].hist(gen_means, bins=30, alpha=0.5, label='Generated')
    axes[0].set_title('Mean Distribution')
    axes[0].legend()
    
    # Std distribution
    real_stds = real_data.std(axis=1).flatten()
    gen_stds = generated_data.std(axis=1).flatten()
    axes[1].hist(real_stds, bins=30, alpha=0.5, label='Real')
    axes[1].hist(gen_stds, bins=30, alpha=0.5, label='Generated')
    axes[1].set_title('Std Distribution')
    axes[1].legend()
    
    # Value distribution
    axes[2].hist(real_data.flatten(), bins=50, alpha=0.5, label='Real', density=True)
    axes[2].hist(generated_data.flatten(), bins=50, alpha=0.5, label='Generated', density=True)
    axes[2].set_title('Value Distribution')
    axes[2].legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def plot_training_curves(log_file, save_path=None):
    """Plot training curves from log file"""
    # Parse log file
    train_losses = []
    val_losses = []
    epochs = []
    
    with open(log_file, 'r') as f:
        for line in f:
            if 'train/loss' in line:
                loss = float(line.split('=')[1].strip())
                train_losses.append(loss)
            elif 'val/loss' in line:
                loss = float(line.split('=')[1].strip())
                val_losses.append(loss)
                epochs.append(len(val_losses))
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss', alpha=0.7)
    if val_losses:
        plt.plot(np.linspace(0, len(train_losses), len(val_losses)), 
                val_losses, label='Val Loss', linewidth=2)
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('Training Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()
