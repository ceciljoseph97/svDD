import os
import argparse
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)


def save_grid(tensors, nrow, save_path, title=None):
    tensors = [t.detach().cpu() for t in tensors]
    num = len(tensors)
    ncol = math.ceil(num / nrow)
    fig, axes = plt.subplots(ncol, nrow, figsize=(nrow * 2, ncol * 2))
    if title:
        fig.suptitle(title)
    axes = np.array(axes).reshape(ncol, nrow)
    for i in range(ncol):
        for j in range(nrow):
            k = i * nrow + j
            axes[i, j].axis('off')
            if k < num:
                img = tensors[k]
                if img.dim() == 3 and img.size(0) == 1:
                    axes[i, j].imshow(img.squeeze(0), cmap='gray')
                elif img.dim() == 3 and img.size(0) == 3:
                    axes[i, j].imshow(img.permute(1, 2, 0))
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def _center_mask(size=28, frac=0.5):
    m = torch.ones(1, size, size)
    w = int(size * frac / 2)
    c = size // 2
    m[:, c - w:c + w, c - w:c + w] = 0.0
    return m

def partial_recon_gen_test(model, test_loader, out_dir, epoch, k=8, mask_frac=0.5):
    os.makedirs(out_dir, exist_ok=True)
    model.eval()
    with torch.no_grad():
        x, _ = next(iter(test_loader))
        x = x[:k].to(DEVICE)
        size = x.size(-1)
        mask = _center_mask(size=size, frac=mask_frac).to(x.device)
        x_mask = x * mask
        recon, _, _ = model(x_mask)

        # Save triptych: original | masked | reconstruction
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(3, k, figsize=(2*k, 6))
        for i in range(k):
            axes[0, i].axis('off'); axes[1, i].axis('off'); axes[2, i].axis('off')
            axes[0, i].imshow(x[i].squeeze(0).detach().cpu(), cmap='gray')
            axes[1, i].imshow(x_mask[i].squeeze(0).detach().cpu(), cmap='gray')
            axes[2, i].imshow(recon[i].squeeze(0).detach().cpu(), cmap='gray')
        axes[0, 0].set_ylabel('orig'); axes[1, 0].set_ylabel('masked'); axes[2, 0].set_ylabel('recon')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'partial_epoch_{epoch:03d}.png'), dpi=150, bbox_inches='tight')
        plt.close()

        # Also save contemporaneous generations for comparison
        z = torch.randn(k, model.encoder.mu.out_features, device=DEVICE)
        gen = model.decoder(z)
        save_grid(list(gen), k, os.path.join(out_dir, f'samples_epoch_{epoch:03d}.png'), title=f'Samples e{epoch}')


class Encoder(nn.Module):
    def __init__(self, in_ch=1, latent_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, 2, 1),  # 28 -> 14
            nn.ReLU(True),
            nn.Conv2d(32, 64, 3, 2, 1),     # 14 -> 7
            nn.ReLU(True),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(True),
        )
        self.mu = nn.Linear(128 * 7 * 7, latent_dim)
        self.logvar = nn.Linear(128 * 7 * 7, latent_dim)

    def forward(self, x):
        h = self.net(x)
        h = h.view(h.size(0), -1)
        return self.mu(h), self.logvar(h)


class Decoder(nn.Module):
    def __init__(self, out_ch=1, latent_dim=32):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 128 * 7 * 7)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),  # 7 -> 14
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),   # 14 -> 28
            nn.ReLU(True),
            nn.Conv2d(32, out_ch, 3, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        h = self.fc(z).view(z.size(0), 128, 7, 7)
        return self.net(h)


class CVAE(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, latent_dim=32):
        super().__init__()
        self.encoder = Encoder(in_ch, latent_dim)
        self.decoder = Decoder(out_ch, latent_dim)
        # For class-conditional generation (optional)
        self.class_embedding = nn.Embedding(10, latent_dim)  # MNIST has 10 classes
        self.use_class_cond = False

    def reparameterize(self, mu, logvar):
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        xhat = self.decoder(z)
        return xhat, mu, logvar

    def sample_class(self, class_label, num_samples=1, device=None):
        """Generate samples conditioned on class label"""
        if device is None:
            device = next(self.parameters()).device
        self.eval()
        with torch.no_grad():
            # Option 1: Pure random + class embedding
            z_random = torch.randn(num_samples, self.encoder.mu.out_features, device=device)
            if self.use_class_cond:
                class_emb = self.class_embedding(torch.tensor([class_label] * num_samples, device=device))
                z = z_random + 0.3 * class_emb  # Mix random and class info
            else:
                z = z_random
            return self.decoder(z)

    def sample_from_latent(self, z):
        """Generate from given latent vector"""
        self.eval()
        with torch.no_grad():
            return self.decoder(z)


def vae_loss(x, xhat, mu, logvar, beta=1.0):
    recon = F.binary_cross_entropy(xhat, x, reduction='sum') / x.size(0)
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    return recon + beta * kld, recon, kld


def get_data(batch_size, dataset_path=None):
    if dataset_path is None or dataset_path == 'mnist':
        transform = transforms.ToTensor()
        train = datasets.MNIST(root='Data/MNIST_raw', train=True, transform=transform, download=True)
        test = datasets.MNIST(root='Data/MNIST_raw', train=False, transform=transform, download=True)
        in_ch = out_ch = 1
    else:
        transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
        ])
        train = datasets.ImageFolder(dataset_path, transform=transform)
        test = datasets.ImageFolder(dataset_path, transform=transform)
        in_ch = out_ch = 1

    # If using folder dataset (e.g., MNIST_processed), take a balanced 200 per class initially
    if dataset_path is not None and dataset_path != 'mnist':
        from collections import defaultdict
        from torch.utils.data import Subset
        rng = random.Random(SEED)

        def balanced_indices(dataset, max_per_class=200):
            # dataset.samples -> list of (path, class_index)
            by_class = defaultdict(list)
            for idx, (_, cls) in enumerate(dataset.samples):
                by_class[int(cls)].append(idx)
            indices = []
            for cls, inds in by_class.items():
                rng.shuffle(inds)
                take = min(max_per_class, len(inds))
                indices.extend(inds[:take])
            rng.shuffle(indices)
            return indices

        train = Subset(train, balanced_indices(train, 200))
        test = Subset(test, balanced_indices(test, 200))

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, test_loader, in_ch, out_ch


def _collect_latents(model, data_loader, limit=2000):
    model.eval()
    zs, ys = [], []
    with torch.no_grad():
        seen = 0
        for x, y in data_loader:
            x = x.to(DEVICE)
            mu, _ = model.encoder(x)
            zs.append(mu.cpu())
            ys.append(y)
            seen += x.size(0)
            if seen >= limit:
                break
    Z = torch.cat(zs)[:limit]
    Y = torch.cat(ys)[:limit]
    return Z, Y


def _save_latents_epoch(out_dir, epoch, Z, Y):
    os.makedirs(os.path.join(out_dir, 'latents'), exist_ok=True)
    torch.save({'z': Z, 'y': Y}, os.path.join(out_dir, 'latents', f'epoch_{epoch:03d}.pt'))


def _save_class_samples(model, out_dir, num_per_class=4):
    """Generate and save samples for each class"""
    model.eval()
    with torch.no_grad():
        for class_id in range(10):
            samples = model.sample_class(class_id, num_per_class, DEVICE)
            save_grid(list(samples), num_per_class, 
                     os.path.join(out_dir, f'class_{class_id}_samples.png'), 
                     title=f'Class {class_id} samples')

def _save_interpolation(model, test_loader, out_dir, a=4, b=5, steps=11):
    # Find one sample for class a and class b
    xa, xb = None, None
    ya, yb = None, None
    with torch.no_grad():
        for x, y in test_loader:
            for i in range(x.size(0)):
                if xa is None and int(y[i]) == int(a):
                    xa = x[i:i+1].to(DEVICE)
                    ya = y[i].item()
                if xb is None and int(y[i]) == int(b):
                    xb = x[i:i+1].to(DEVICE)
                    yb = y[i].item()
                if xa is not None and xb is not None:
                    break
            if xa is not None and xb is not None:
                break
        if xa is None or xb is None:
            print(f"Interpolation skipped: could not find classes {a} and {b} in test set")
            return
        mu_a, _ = model.encoder(xa)
        mu_b, _ = model.encoder(xb)
        imgs = []
        for t in np.linspace(0.0, 1.0, steps):
            z = (1 - t) * mu_a + t * mu_b
            img = model.decoder(z.to(DEVICE))[0].detach().cpu()
            imgs.append(img)
        save_grid(imgs, steps, os.path.join(out_dir, f'interpolate_{a}_to_{b}.png'), title=f'{a} → {b}')


def train(model, train_loader, test_loader, epochs, lr, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    # Resume if checkpoint exists
    ckpt_path = os.path.join(out_dir, 'checkpoint_latest.pth')
    start_epoch = 1
    if os.path.exists(ckpt_path):
        try:
            ckpt = torch.load(ckpt_path, map_location=DEVICE)
            model.load_state_dict(ckpt['model_state'], strict=False)
            opt.load_state_dict(ckpt['optimizer_state'])
            start_epoch = int(ckpt.get('epoch', 0)) + 1
            print(f"Resuming from epoch {start_epoch-1}")
        except Exception as e:
            print(f"Resume failed, starting fresh: {e}")

    history = []
    for ep in range(start_epoch, epochs + 1):
        model.train()
        tot, tot_r, tot_k = 0.0, 0.0, 0.0
        for x, _ in train_loader:
            x = x.to(DEVICE)
            opt.zero_grad()
            xhat, mu, logvar = model(x)
            beta = min(1.0, ep / 20.0)
            loss, r, k = vae_loss(x, xhat, mu, logvar, beta=beta)
            loss.backward()
            opt.step()
            tot += loss.item()
            tot_r += r.item()
            tot_k += k.item()

        # eval
        model.eval()
        with torch.no_grad():
            x, _ = next(iter(test_loader))
            x = x[:16].to(DEVICE)
            xhat, _, _ = model(x)
            save_grid(list(x), 4, os.path.join(out_dir, f"ep{ep:03d}_inputs.png"), title=f"Inputs e{ep}")
            save_grid(list(xhat), 4, os.path.join(out_dir, f"ep{ep:03d}_recons.png"), title=f"Recons e{ep}")

        # Save latents for this epoch
        Z, Y = _collect_latents(model, test_loader, limit=2000)
        _save_latents_epoch(out_dir, ep, Z, Y)

        # Save latest checkpoint for resume
        torch.save({'model_state': model.state_dict(), 'optimizer_state': opt.state_dict(), 'epoch': ep}, ckpt_path)

        # Every 10 epochs: partial reconstruction + generation snapshot
        if ep % 10 == 0:
            partial_recon_gen_test(model, test_loader, out_dir, ep, k=8, mask_frac=0.5)

        history.append((tot / len(train_loader), tot_r / len(train_loader), tot_k / len(train_loader)))
        print(f"Epoch {ep:03d} loss {history[-1][0]:.4f} recon {history[-1][1]:.4f} kl {history[-1][2]:.4f}")

    # sample
    with torch.no_grad():
        z = torch.randn(16, model.encoder.mu.out_features, device=DEVICE)
        samples = model.decoder(z)
        save_grid(list(samples), 4, os.path.join(out_dir, f"samples.png"), title="Samples")
        # also save class interpolation 4->5 by default
        _save_interpolation(model, test_loader, out_dir, a=4, b=5, steps=11)
        # Save class-specific samples
        _save_class_samples(model, out_dir, num_per_class=4)

    # simple t-SNE on latents
    try:
        from sklearn.manifold import TSNE
        zs, ys = [], []
        imgs = []
        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(DEVICE)
                mu, logvar = model.encoder(x)
                zs.append(mu.cpu())
                ys.append(y)
                imgs.append(x.cpu())
        Z = torch.cat(zs)[:2000].numpy()
        Y = torch.cat(ys)[:2000].numpy()
        IM = torch.cat(imgs)[:2000].cpu()
        try:
            tsne = TSNE(n_components=2, random_state=42, n_iter=1000)
        except TypeError:
            tsne = TSNE(n_components=2, random_state=42)
        Z2 = tsne.fit_transform(Z)
        plt.figure(figsize=(8, 6))
        for d in range(10):
            m = Y == d
            if m.sum():
                plt.scatter(Z2[m, 0], Z2[m, 1], s=5, label=str(d))
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'tsne.png'), dpi=150)
        plt.close()

        # t-SNE with thumbnail images (subset for clarity/perf)
        try:
            import matplotlib.offsetbox as offsetbox
            rng = np.random.RandomState(SEED)
            n = min(500, len(Z2))
            idx = rng.choice(len(Z2), size=n, replace=False)
            Zs = Z2[idx]
            IMs = IM[idx]
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.set_title('t-SNE with images')
            ax.scatter(Z2[:, 0], Z2[:, 1], s=2, c='lightgray')
            for i in range(len(Zs)):
                img = IMs[i]
                # ensure HxW
                if img.dim() == 3 and img.size(0) == 1:
                    thumb = img.squeeze(0).numpy()
                    cmap = 'gray'
                else:
                    thumb = img.permute(1, 2, 0).numpy()
                    cmap = None
                thumb = np.clip(thumb, 0, 1)
                oi = offsetbox.OffsetImage(thumb, zoom=0.5, cmap=cmap)
                ab = offsetbox.AnnotationBbox(oi, (Zs[i, 0], Zs[i, 1]), frameon=False)
                ax.add_artist(ab)
            ax.set_xticks([]); ax.set_yticks([])
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, 'tsne_images.png'), dpi=150)
            plt.close()
        except Exception as e:
            print(f"t-SNE with images skipped: {e}")
    except Exception as e:
        print(f"TSNE skipped: {e}")

    # Save full checkpoint for retraining
    ckpt = {
        'model_state': model.state_dict(),
        'optimizer_state': opt.state_dict(),
        'epochs': epochs,
        'latent_dim': model.encoder.mu.out_features,
    }
    torch.save(ckpt, os.path.join(out_dir, 'checkpoint_final.pth'))

    return history


def load_model(model_path, out_dir):
    """Load model from checkpoint or model file"""
    model = CVAE(in_ch=1, out_ch=1, latent_dim=32).to(DEVICE)
    
    # Try to load checkpoint
    if os.path.exists(os.path.join(out_dir, 'checkpoint_final.pth')):
        ckpt = torch.load(os.path.join(out_dir, 'checkpoint_final.pth'), map_location=DEVICE)
        model.load_state_dict(ckpt['model_state'], strict=False)
        print(f"Loaded model from {out_dir}/checkpoint_final.pth")
    elif os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=DEVICE), strict=False)
        print(f"Loaded model from {model_path}")
    else:
        print(f"No model found at {model_path} or {out_dir}/checkpoint_final.pth")
        return None
    return model


def generate_samples(model_path, out_dir, class_id=None, num_samples=16):
    """Generate samples from trained model"""
    model = load_model(model_path, out_dir)
    if model is None:
        return
    
    os.makedirs(out_dir, exist_ok=True)
    
    if class_id is not None:
        # Generate specific class
        samples = model.sample_class(class_id, num_samples, DEVICE)
        save_grid(list(samples), 4, os.path.join(out_dir, f'generated_class_{class_id}.png'), 
                 title=f'Generated Class {class_id}')
        print(f"Saved {num_samples} samples of class {class_id}")
    else:
        # Generate random samples
        samples = model.sample_class(0, num_samples, DEVICE)  # Will use random since use_class_cond=False
        save_grid(list(samples), 4, os.path.join(out_dir, 'generated_random.png'), 
                 title='Generated Random Samples')
        print(f"Saved {num_samples} random samples")


def compute_anomaly_scores(model, data_loader, max_items=5000, divergence="bce", beta_b=1.5):
    """Compute simple anomaly scores for a trained VAE.

    Returns a dict with per-sample scores: recon_error, kl_term, total_elbo, latent_norm, labels.
    divergence: 'bce' | 'mse' | 'beta' (beta-divergence with parameter beta_b)
    """
    model.eval()
    scores = {
        'recon_error': [],
        'kl_term': [],
        'total_elbo': [],
        'latent_norm': [],
        'labels': [],
    }
    eps = 1e-6
    seen = 0
    with torch.no_grad():
        for x, y in data_loader:
            x = x.to(DEVICE)
            xhat, mu, logvar = model(x)
            if divergence == 'mse':
                recon = F.mse_loss(xhat, x, reduction='none').flatten(1).sum(dim=1)
            elif divergence == 'beta':
                # beta-divergence per pixel; clamp to avoid log/zero issues
                b = beta_b
                x_pos = torch.clamp(x, min=0.0, max=1.0)
                y_pos = torch.clamp(xhat, min=eps, max=1.0)
                # D_b(x||y) = 1/(b(b-1)) * (x^b + (b-1) y^b - b x y^{b-1})
                term = (x_pos.pow(b) + (b - 1.0) * y_pos.pow(b) - b * x_pos * y_pos.pow(b - 1.0)) / (b * (b - 1.0))
                recon = term.flatten(1).sum(dim=1)
            else:
                # BCE
                recon = F.binary_cross_entropy(xhat, x, reduction='none').flatten(1).sum(dim=1)

            kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
            total = recon + kld
            z_norm = mu.norm(p=2, dim=1)

            scores['recon_error'].extend(recon.detach().cpu().tolist())
            scores['kl_term'].extend(kld.detach().cpu().tolist())
            scores['total_elbo'].extend(total.detach().cpu().tolist())
            scores['latent_norm'].extend(z_norm.detach().cpu().tolist())
            scores['labels'].extend(y.detach().cpu().tolist())

            seen += x.size(0)
            if seen >= max_items:
                break

    return scores

def run_interpolation(model_path, out_dir, class_a=4, class_b=5, steps=11):
    """Generate interpolation between two classes"""
    model = load_model(model_path, out_dir)
    if model is None:
        return
    
    os.makedirs(out_dir, exist_ok=True)
    
    # Load test data to find samples
    train_loader, test_loader, _, _ = get_data(128, 'Data/MNIST_processed')
    
    _save_interpolation(model, test_loader, out_dir, class_a, class_b, steps)
    print(f"Saved interpolation from class {class_a} to {class_b} with {steps} steps")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--batch', type=int, default=128)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--latent', type=int, default=32)
    p.add_argument('--dataset', type=str, default='Data/MNIST_processed', help='mnist or path to ImageFolder')
    p.add_argument('--out', type=str, default='CVAE_Results')
    
    # Mode flags
    mode_group = p.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--train', action='store_true', help='Train the model')
    mode_group.add_argument('--infer', action='store_true', help='Run inference on test set')
    mode_group.add_argument('--generate', action='store_true', help='Generate new samples')
    mode_group.add_argument('--interpolate', action='store_true', help='Generate interpolation between two classes')
    
    # Generation-specific args
    p.add_argument('--class_id', type=int, help='Class ID for generation (0-9 for MNIST)')
    p.add_argument('--num_samples', type=int, default=16, help='Number of samples to generate')
    p.add_argument('--model_path', type=str, help='Path to model file for inference/generation')
    
    # Interpolation-specific args
    p.add_argument('--class_a', type=int, default=4, help='First class for interpolation (0-9)')
    p.add_argument('--class_b', type=int, default=5, help='Second class for interpolation (0-9)')
    p.add_argument('--steps', type=int, default=11, help='Number of interpolation steps')
    
    args = p.parse_args()

    print('Device:', DEVICE)
    
    if args.train:
        print("=== TRAINING MODE ===")
        train_loader, test_loader, in_ch, out_ch = get_data(args.batch, None if args.dataset == 'mnist' else args.dataset)
        model = CVAE(in_ch=in_ch, out_ch=out_ch, latent_dim=args.latent).to(DEVICE)
        os.makedirs(args.out, exist_ok=True)
        hist = train(model, train_loader, test_loader, args.epochs, args.lr, args.out)
        print(f"Training completed. Model saved to {args.out}")
        
    elif args.infer:
        print("=== INFERENCE MODE ===")
        if not args.model_path:
            args.model_path = os.path.join(args.out, 'checkpoint_final.pth')
        train_loader, test_loader, in_ch, out_ch = get_data(args.batch, None if args.dataset == 'mnist' else args.dataset)
        model = CVAE(in_ch=in_ch, out_ch=out_ch, latent_dim=args.latent).to(DEVICE)
        
        # Load model
        if os.path.exists(args.model_path):
            ckpt = torch.load(args.model_path, map_location=DEVICE)
            model.load_state_dict(ckpt['model_state'])
            print(f"Loaded model from {args.model_path}")
        else:
            print(f"No model found at {args.model_path}")
            return
            
        # Run inference
        model.eval()
        with torch.no_grad():
            x, _ = next(iter(test_loader))
            x = x[:16].to(DEVICE)
            xhat, _, _ = model(x)
            save_grid(list(x), 4, os.path.join(args.out, 'inference_inputs.png'), title='Inference Inputs')
            save_grid(list(xhat), 4, os.path.join(args.out, 'inference_recons.png'), title='Inference Reconstructions')
        print(f"Inference completed. Results saved to {args.out}")
        
    elif args.generate:
        print("=== GENERATION MODE ===")
        model_path = args.model_path or os.path.join(args.out, 'cvae.pth')
        generate_samples(model_path, args.out, args.class_id, args.num_samples)
        
    elif args.interpolate:
        print("=== INTERPOLATION MODE ===")
        model_path = args.model_path or os.path.join(args.out, 'cvae.pth')
        run_interpolation(model_path, args.out, args.class_a, args.class_b, args.steps)


if __name__ == '__main__':
    main()


