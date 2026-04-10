"""
Create a GIF showing latent space evolution over training epochs
"""

import os
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from sklearn.manifold import TSNE
import argparse

def load_latent_files(latents_dir):
    """Load all latent files and sort by epoch"""
    latent_files = glob.glob(os.path.join(latents_dir, "epoch_*.pt"))
    latent_files.sort(key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))
    
    epochs = []
    latents = []
    labels = []
    
    for file_path in latent_files:
        epoch_num = int(os.path.basename(file_path).split('_')[1].split('.')[0])
        data = torch.load(file_path, map_location='cpu')
        epochs.append(epoch_num)
        latents.append(data['z'].numpy())
        labels.append(data['y'].numpy())
    
    return epochs, latents, labels

def create_latent_gif(latents_dir, output_path, max_samples=1000, fps=2, duration_ms=500, max_duration_s=20):
    """Create GIF showing latent space evolution"""
    
    print("Loading latent files...")
    epochs, latents_list, labels_list = load_latent_files(latents_dir)
    
    if not epochs:
        print("No latent files found!")
        return
    
    print(f"Found {len(epochs)} epochs: {min(epochs)} to {max(epochs)}")
    
    # Subsample data for performance
    sample_indices = np.random.choice(len(latents_list[0]), 
                                    size=min(max_samples, len(latents_list[0])), 
                                    replace=False)
    
    # Prepare data
    all_latents = []
    all_labels = []
    for latents, labels in zip(latents_list, labels_list):
        all_latents.append(latents[sample_indices])
        all_labels.append(labels[sample_indices])
    
    # Compute t-SNE for all epochs
    print("Computing t-SNE for all epochs...")
    tsne_results = []
    
    for i, (latents, epoch) in enumerate(zip(all_latents, epochs)):
        print(f"Processing epoch {epoch} ({i+1}/{len(epochs)})")
        try:
            tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=300)
            tsne_2d = tsne.fit_transform(latents)
            tsne_results.append(tsne_2d)
        except:
            # Fallback for older sklearn versions
            try:
                tsne = TSNE(n_components=2, random_state=42, n_iter=300)
                tsne_2d = tsne.fit_transform(latents)
                tsne_results.append(tsne_2d)
            except:
                print(f"Failed to compute t-SNE for epoch {epoch}")
                continue
    
    if not tsne_results:
        print("Failed to compute t-SNE for any epoch!")
        return
    
    # Create animation
    print("Creating animation...")
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Get consistent axis limits
    all_points = np.vstack(tsne_results)
    x_min, x_max = all_points[:, 0].min(), all_points[:, 0].max()
    y_min, y_max = all_points[:, 1].min(), all_points[:, 1].max()
    
    # Add some padding
    x_range = x_max - x_min
    y_range = y_max - y_min
    x_min -= 0.1 * x_range
    x_max += 0.1 * x_range
    y_min -= 0.1 * y_range
    y_max += 0.1 * y_range
    
    def animate(frame):
        ax.clear()
        
        if frame < len(tsne_results):
            tsne_2d = tsne_results[frame]
            epoch = epochs[frame]
            labels = all_labels[frame]
            
            # Plot each class with different colors
            colors = plt.cm.tab10(np.linspace(0, 1, 10))
            for class_id in range(10):
                mask = labels == class_id
                if mask.sum() > 0:
                    ax.scatter(tsne_2d[mask, 0], tsne_2d[mask, 1], 
                             c=[colors[class_id]], label=f'Class {class_id}', 
                             s=20, alpha=0.7)
            
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_title(f'Latent Space Evolution - Epoch {epoch}', fontsize=16)
            ax.set_xlabel('t-SNE Dimension 1')
            ax.set_ylabel('t-SNE Dimension 2')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
            ax.grid(True, alpha=0.3)
    
    # Create animation
    anim = FuncAnimation(fig, animate, frames=len(tsne_results), 
                        interval=duration_ms, repeat=True, blit=False)
    
    # Save as GIF (cap total duration to max_duration_s)
    total_frames = len(tsne_results)
    computed_fps = max(1, int(np.ceil(total_frames / max_duration_s)))
    if fps != computed_fps:
        print(f"Adjusting fps from {fps} to {computed_fps} to keep duration ≤ {max_duration_s}s")
    print(f"Saving GIF to {output_path}...")
    writer = PillowWriter(fps=computed_fps)
    anim.save(output_path, writer=writer)
    
    plt.close()
    print(f"GIF saved successfully!")
    print(f"Epochs: {epochs[0]} to {epochs[-1]}")
    print(f"Frames: {len(tsne_results)}")
    print(f"Frames: {total_frames}, FPS: {computed_fps}, Duration ≈ {total_frames / computed_fps:.1f} seconds")

def create_latent_heatmap_gif(latents_dir, output_path, max_samples=1000, fps=2, max_duration_s=20,
                              bins=80, smooth_sigma=0.8, adaptive=True, log_scale=False):
    """Create GIF showing latent space as heatmap over time.
    - bins: number of bins per axis
    - smooth_sigma: gaussian blur sigma (0 disables)
    - adaptive: adjust color scale per frame (EMA) to reveal changes
    - log_scale: use logarithmic color normalization
    """
    print("Loading latent files for heatmap...")
    epochs, latents_list, _ = load_latent_files(latents_dir)
    if not epochs:
        print("No latent files found!")
        return

    # Deterministic subsample for consistent point identity
    rng = np.random.RandomState(42)
    sample_indices = rng.choice(len(latents_list[0]), size=min(max_samples, len(latents_list[0])), replace=False)

    # Prepare per-frame latents
    all_latents = [lat[sample_indices] for lat in latents_list]

    # Fix axes and color scale across frames
    concat = np.concatenate(all_latents, axis=0)
    x_min, x_max = np.percentile(concat[:, 0], [1, 99])
    y_min, y_max = np.percentile(concat[:, 1], [1, 99])
    pad_x = 0.05 * (x_max - x_min)
    pad_y = 0.05 * (y_max - y_min)
    x_min, x_max = x_min - pad_x, x_max + pad_x
    y_min, y_max = y_min - pad_y, y_max + pad_y
    xedges = np.linspace(x_min, x_max, bins + 1)
    yedges = np.linspace(y_min, y_max, bins + 1)

    vmax = 0
    for lat in all_latents:
        hist, _, _ = np.histogram2d(lat[:, 0], lat[:, 1], bins=[xedges, yedges])
        if smooth_sigma and smooth_sigma > 0:
            try:
                from scipy.ndimage import gaussian_filter
                hist = gaussian_filter(hist, sigma=smooth_sigma)
            except Exception:
                pass
        vmax = max(vmax, float(hist.max()))
    vmin = 0

    # Initialize once
    fig, ax = plt.subplots(figsize=(10, 8))
    hist0, _, _ = np.histogram2d(all_latents[0][:, 0], all_latents[0][:, 1], bins=[xedges, yedges])
    if smooth_sigma and smooth_sigma > 0:
        try:
            from scipy.ndimage import gaussian_filter
            hist0 = gaussian_filter(hist0, sigma=smooth_sigma)
        except Exception:
            pass
    norm = None
    if log_scale:
        from matplotlib.colors import LogNorm
        hist0 = np.maximum(hist0, 1e-6)
        norm = LogNorm(vmin=1e-6, vmax=max(vmax, 1.0))
    im = ax.imshow(
        hist0.T,
        extent=[x_min, x_max, y_min, y_max],
        origin='lower', cmap='viridis', aspect='auto',
        vmin=None if log_scale else vmin, vmax=None if log_scale else vmax,
        norm=norm,
    )
    cbar = fig.colorbar(im, ax=ax, label='Count')
    ax.set_title(f'Latent Space Density - Epoch {epochs[0]}', fontsize=16)
    ax.set_xlabel('Latent Dimension 1')
    ax.set_ylabel('Latent Dimension 2')

    vmax_ema = vmax
    def animate(frame):
        if frame >= len(all_latents):
            return [im]
        epoch = epochs[frame]
        latents = all_latents[frame]
        hist, _, _ = np.histogram2d(latents[:, 0], latents[:, 1], bins=[xedges, yedges])
        if smooth_sigma and smooth_sigma > 0:
            try:
                from scipy.ndimage import gaussian_filter
                hist = gaussian_filter(hist, sigma=smooth_sigma)
            except Exception:
                pass
        if log_scale:
            hist = np.maximum(hist, 1e-6)
        im.set_data(hist.T)
        if adaptive and not log_scale:
            nonlocal vmax_ema
            vmax_frame = float(np.percentile(hist, 99))
            vmax_ema = 0.9 * vmax_ema + 0.1 * vmax_frame
            im.set_clim(vmin, max(vmax_ema, 1e-6))
            cbar.update_normal(im)
        elif log_scale:
            from matplotlib.colors import LogNorm
            vmax_frame = float(np.percentile(hist, 99))
            im.set_norm(LogNorm(vmin=1e-6, vmax=max(vmax_frame, 1.0)))
            cbar.update_normal(im)
        ax.set_title(f'Latent Space Density - Epoch {epoch}', fontsize=16)
        return [im]

    anim = FuncAnimation(fig, animate, frames=len(all_latents), interval=500, repeat=True, blit=True)
    
    # Save as GIF with duration cap
    total_frames = len(all_latents)
    computed_fps = max(1, int(np.ceil(total_frames / max_duration_s)))
    if fps != computed_fps:
        print(f"Adjusting fps from {fps} to {computed_fps} to keep duration ≤ {max_duration_s}s")
    print(f"Saving heatmap GIF to {output_path}...")
    writer = PillowWriter(fps=computed_fps)
    anim.save(output_path, writer=writer)
    
    plt.close()
    print(f"Heatmap GIF saved successfully!")

def main():
    parser = argparse.ArgumentParser(description='Create latent space evolution GIF')
    parser.add_argument('--latents_dir', type=str, default='CVAE_Results/latents', 
                       help='Directory containing epoch_*.pt files')
    parser.add_argument('--output', type=str, default='latent_evolution.gif',
                       help='Output GIF filename')
    parser.add_argument('--max_samples', type=int, default=1000,
                       help='Maximum samples to use for t-SNE (for performance)')
    parser.add_argument('--fps', type=int, default=2,
                       help='Frames per second for GIF')
    parser.add_argument('--duration_ms', type=int, default=500,
                       help='Duration per frame in milliseconds')
    parser.add_argument('--mode', type=str, default='tsne', choices=['tsne', 'heatmap', 'both'],
                       help='Visualization mode: tsne, heatmap, or both')
    
    args = parser.parse_args()
    
    if args.mode in ['tsne', 'both']:
        output_tsne = args.output.replace('.gif', '_tsne.gif')
        create_latent_gif(args.latents_dir, output_tsne, 
                         args.max_samples, args.fps, args.duration_ms)
    
    if args.mode in ['heatmap', 'both']:
        output_heatmap = args.output.replace('.gif', '_heatmap.gif')
        create_latent_heatmap_gif(args.latents_dir, output_heatmap, 
                                 args.max_samples, args.fps)

if __name__ == '__main__':
    main()
