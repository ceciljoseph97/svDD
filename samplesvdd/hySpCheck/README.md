# hySpCheck: Hyperbolic Multi-Sphere Deep SVDD

This folder contains a hyperbolic (Poincare-ball) multi-class extension of Deep SVDD over MNIST processed digits.

- shared LeNet AE backbone
- per-digit projection heads
- per-digit hyperbolic centers and radii
- soft-boundary or one-class objective
- inter-class exclusion penalty (push sample away from non-true spheres)
- overlap penalty (separate class spheres geometrically)

## Train (CUDA, lsAirsim)

```powershell
& 'C:\Users\cecil\anaconda3\envs\lsAirsim\python.exe' 'D:\Doctorate\EngineCheck\oasplTurbojet\SampleSvDD\samplesvdd\hySpCheck\train_hyp_mnist_multi.py' --mnist_processed_dir 'D:\Doctorate\EngineCheck\oasplTurbojet\SampleSvDD\CVAEChecked\Data\MNIST_processed' --xp_path 'D:\Doctorate\EngineCheck\oasplTurbojet\SampleSvDD\samplesvdd\hySpCheck\runs\hyp_multi_cuda' --device cuda --digits all --objective soft-boundary --curvature 1.0 --lambda_excl 0.01 --margin_excl 0.1 --lambda_overlap 0.01 --margin_overlap 0.05
```

## Eval

```powershell
& 'C:\Users\cecil\anaconda3\envs\lsAirsim\python.exe' 'D:\Doctorate\EngineCheck\oasplTurbojet\SampleSvDD\samplesvdd\hySpCheck\eval_hyp_mnist_multi.py' --mnist_processed_dir 'D:\Doctorate\EngineCheck\oasplTurbojet\SampleSvDD\CVAEChecked\Data\MNIST_processed' --checkpoint_path 'D:\Doctorate\EngineCheck\oasplTurbojet\SampleSvDD\samplesvdd\hySpCheck\runs\hyp_multi_cuda\checkpoint_latest.pth' --device cuda --out_dir 'D:\Doctorate\EngineCheck\oasplTurbojet\SampleSvDD\samplesvdd\hySpCheck\runs\hyp_multi_cuda\eval_test'
```

## Visualize latent t-SNE

```powershell
& 'C:\Users\cecil\anaconda3\envs\lsAirsim\python.exe' 'D:\Doctorate\EngineCheck\oasplTurbojet\SampleSvDD\samplesvdd\hySpCheck\visualize_hyp_mnist_multi.py' --mnist_processed_dir 'D:\Doctorate\EngineCheck\oasplTurbojet\SampleSvDD\CVAEChecked\Data\MNIST_processed' --checkpoint_path 'D:\Doctorate\EngineCheck\oasplTurbojet\SampleSvDD\samplesvdd\hySpCheck\runs\hyp_multi_cuda\checkpoint_latest.pth' --device cuda --split test --max_samples 2000 --perplexity 30 --n_iter 1000 --out_dir 'D:\Doctorate\EngineCheck\oasplTurbojet\SampleSvDD\samplesvdd\hySpCheck\runs\hyp_multi_cuda\tsne'
```

## Visualize Poincare disk (geodesic-aware)

```powershell
& 'C:\Users\cecil\anaconda3\envs\lsAirsim\python.exe' 'D:\Doctorate\EngineCheck\oasplTurbojet\SampleSvDD\samplesvdd\hySpCheck\visualize_hyp_poincare_disk.py' --mnist_processed_dir 'D:\Doctorate\EngineCheck\oasplTurbojet\SampleSvDD\CVAEChecked\Data\MNIST_processed' --checkpoint_path 'D:\Doctorate\EngineCheck\oasplTurbojet\SampleSvDD\samplesvdd\hySpCheck\runs\hyp_multi_cuda\checkpoint_latest.pth' --device cuda --split test --max_samples 1000 --embed_mode auto --out_dir 'D:\Doctorate\EngineCheck\oasplTurbojet\SampleSvDD\samplesvdd\hySpCheck\runs\hyp_multi_cuda\poincare'
```

## Quick smoke (fast CPU)

```powershell
& 'C:\Users\cecil\anaconda3\envs\lsAirsim\python.exe' 'D:\Doctorate\EngineCheck\oasplTurbojet\SampleSvDD\samplesvdd\hySpCheck\train_hyp_mnist_multi.py' --mnist_processed_dir 'D:\Doctorate\EngineCheck\oasplTurbojet\SampleSvDD\CVAEChecked\Data\MNIST_processed' --xp_path 'D:\Doctorate\EngineCheck\oasplTurbojet\SampleSvDD\samplesvdd\hySpCheck\runs\smoke' --device cpu --ae_n_epochs 1 --svdd_n_epochs 1 --max_train_samples 256 --max_test_samples 256 --eval_every 1
```

