# hySpCheckcifar: Hyperbolic coupled multi-sphere for CIFAR10

Same methodology as hySpCheck (MNIST) but for CIFAR-10:

- shared AE backbone
- class-specific hyperbolic spheres
- Ruff-like SVDD term + inter-class exclusion + overlap penalty
- train / eval / t-SNE / interactive 3D Poincare visualization

## Train

```cmd
python D:\Doctorate\EngineCheck\oasplTurbojet\SampleSvDD\samplesvdd\hySpCheckcifar\train_hyp_cifar_multi.py --data_root D:\Doctorate\EngineCheck\oasplTurbojet\SampleSvDD\samplesvdd\hySpCheckcifar\data --xp_path D:\Doctorate\EngineCheck\oasplTurbojet\SampleSvDD\samplesvdd\hySpCheckcifar\runs\hyp_cifar_cuda --device cuda --classes all --objective soft-boundary --curvature 1.0 --lambda_excl 0.01 --margin_excl 0.1 --lambda_overlap 0.01 --margin_overlap 0.05
```

## Eval

```cmd
python D:\Doctorate\EngineCheck\oasplTurbojet\SampleSvDD\samplesvdd\hySpCheckcifar\eval_hyp_cifar_multi.py --data_root D:\Doctorate\EngineCheck\oasplTurbojet\SampleSvDD\samplesvdd\hySpCheckcifar\data --checkpoint_path D:\Doctorate\EngineCheck\oasplTurbojet\SampleSvDD\samplesvdd\hySpCheckcifar\runs\hyp_cifar_cuda\checkpoint_latest.pth --device cuda --out_dir D:\Doctorate\EngineCheck\oasplTurbojet\SampleSvDD\samplesvdd\hySpCheckcifar\runs\hyp_cifar_cuda\eval_test
```

## Interactive 3D Poincare walk

```cmd
python D:\Doctorate\EngineCheck\oasplTurbojet\SampleSvDD\samplesvdd\hySpCheckcifar\visualize_hyp_cifar_poincare_3d_interactive.py --data_root D:\Doctorate\EngineCheck\oasplTurbojet\SampleSvDD\samplesvdd\hySpCheckcifar\data --checkpoint_path D:\Doctorate\EngineCheck\oasplTurbojet\SampleSvDD\samplesvdd\hySpCheckcifar\runs\hyp_cifar_cuda\checkpoint_latest.pth --device cuda --max_samples 1200 --embed_mode auto --out_dir D:\Doctorate\EngineCheck\oasplTurbojet\SampleSvDD\samplesvdd\hySpCheckcifar\runs\hyp_cifar_cuda\poincare3d
```

## t-SNE (class names in legend)

```cmd
python D:\Doctorate\EngineCheck\oasplTurbojet\SampleSvDD\samplesvdd\hySpCheckcifar\visualize_hyp_cifar_tsne.py --data_root D:\Doctorate\EngineCheck\oasplTurbojet\SampleSvDD\samplesvdd\hySpCheckcifar\data --checkpoint_path D:\Doctorate\EngineCheck\oasplTurbojet\SampleSvDD\samplesvdd\hySpCheckcifar\runs\hyp_cifar_cuda\checkpoint_latest.pth --device cuda --max_samples 2000 --perplexity 30 --n_iter 1000 --out_dir D:\Doctorate\EngineCheck\oasplTurbojet\SampleSvDD\samplesvdd\hySpCheckcifar\runs\hyp_cifar_cuda\tsne
```

## Quick smoke

```cmd
python D:\Doctorate\EngineCheck\oasplTurbojet\SampleSvDD\samplesvdd\hySpCheckcifar\train_hyp_cifar_multi.py --data_root D:\Doctorate\EngineCheck\oasplTurbojet\SampleSvDD\samplesvdd\hySpCheckcifar\data --xp_path D:\Doctorate\EngineCheck\oasplTurbojet\SampleSvDD\samplesvdd\hySpCheckcifar\runs\smoke --device cpu --ae_n_epochs 1 --svdd_n_epochs 1 --max_train_samples 512 --max_test_samples 512 --eval_every 1
```

