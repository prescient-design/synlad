# SLURM job templates

Example SLURM submission scripts for training on a GPU cluster. They are templates: replace the placeholder shell
variables (`${WORK_DIR}`, `${WANDB_DIR}`, `${ACCOUNT}`, `${PARTITION}`, etc.)
with values appropriate for your cluster, or export them in your environment
before submitting.

Note: `#SBATCH` directives are parsed by SLURM before the shell runs, so shell
variables in them (e.g. `${LOG_DIR}`) are **not** expanded. Set log paths via
SLURM's environment variables before `sbatch`:

```bash
export SBATCH_OUTPUT=$LOG_DIR/train_autoencoder_%j.out
export SBATCH_ERROR=$LOG_DIR/train_autoencoder_%j.err
sbatch train_autoencoder.slurm
```

Set `WANDB_API_KEY` in your shell (or `~/.netrc`) before submitting.

## Available templates

- `train_autoencoder.slurm`        — single-GPU autoencoder training.
- `train_diffusion.slurm`          — single-GPU latent diffusion training.
