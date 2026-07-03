# SynLaD: Latent Diffusion for Generating Synthesizable Molecules Conditioned on 3D Pharmacophore Profiles

Implementation of SynLaD, a generative model for pharmacophore-conditioned small-molecule design in
synthesisable space. You can find the paper on [arxiv](https://arxiv.org/abs/2607.01105).

## Installation

The project is managed with uv. Install uv first
(you can find installation instructions [here](https://docs.astral.sh/uv/)), then from the repository
root:

```bash
uv sync                                              # core deps + editable install of synlad
uv sync --extra dev                                  # add dev tooling
uv sync --extra dev --extra openeye                  # also install OpenEye toolkits (requires license)
uv sync --extra dev --extra openeye --extra retro    # add retrosynthesis stack (aizynthfinder + reaction-utils)
```

This creates `.venv/` pinned to Python 3.12 with the exact versions in
`uv.lock`. Activate it with `source .venv/bin/activate`.

Notes on the dependency stack:

- OpenEye toolkits (used for ROCS shape/colour scoring and conformer
  generation) are not on public PyPI and require a valid license. They are
  pulled from OpenEye's anaconda.org index when you pass `--extra openeye`
  to `uv sync`. See <https://docs.eyesopen.com/toolkits/python/> for licence
  setup.

## Data

The dataset is hosted on [Zenodo](https://zenodo.org/records/20945682). Create
a `data/` directory at the repository root and download the files into it:

```bash
mkdir data && cd data
curl -s https://zenodo.org/api/records/20945682 \
  | jq -r '.files[].links.self' \
  | wget --content-disposition -i -
```

The archive already contains the train/val/test splits and the pre-processed
auxiliary files (e.g. training SMILES, pharmacophore references) needed for
both training and inference, so no further preprocessing is required.

## Configuration

Runtime paths are configured through Hydra. The most useful environment
variables (with sensible defaults defined in `configs/paths/default.yaml`) are:

| Variable           | Purpose                                |
| ------------------ | -------------------------------------- |
| `PROJECT_ROOT`     | Repository root (auto-detected)        |
| `DATA_DIR`         | Pre-processed datasets                 |
| `OUTPUT_DIR`       | Run outputs (logs, checkpoints, plots) |
| `CHECKPOINTS_DIR`  | Override for model checkpoints         |
| `WANDB_PROJECT`    | Weights & Biases project               |
| `WANDB_ENTITY`     | Weights & Biases entity                |
| `WANDB_API_KEY`    | Weights & Biases credential            |


## Training

Training has two stages: first the 3D autoencoder (VAE), then the latent
diffusion model that operates in the VAE's latent space.

**1. Train the 3D autoencoder.** Uses [configs/experiment/train_uspto.yaml](configs/experiment/train_uspto.yaml):

```bash
python -m synlad.train_autoencoder experiment=train_uspto
```

The checkpoint directory it produces is what you pass to the diffusion model
via `diffusion_module.autoencoder_ckpt=...` in the next step.

**2. Train the latent diffusion model.** Two experiment configs are
provided, depending on whether you want the diffusion model to be
pharmacophore-conditioned:

```bash
# Unconditional latent diffusion
python -m synlad.train_diffusion experiment=train_diffusion_uncond \
    diffusion_module.autoencoder_ckpt=/path/to/vae.ckpt

# Pharmacophore-conditioned latent diffusion
python -m synlad.train_diffusion experiment=train_diffusion_ph4_cond \
    diffusion_module.autoencoder_ckpt=/path/to/vae.ckpt
```

See [configs/experiment/train_diffusion_uncond.yaml](configs/experiment/train_diffusion_uncond.yaml)
and [configs/experiment/train_diffusion_ph4_cond.yaml](configs/experiment/train_diffusion_ph4_cond.yaml)
for the differences (`use_conditioning`, `do_pharmacophores`, dropout, etc.).

Reference SLURM templates for both stages live under
[examples/slurm/](examples/slurm/).

## Evaluating the autoencoder

[src/synlad/eval_autoencoder.py](src/synlad/eval_autoencoder.py) reconstructs
a set of molecules from the VAE's latent space and reports
reconstruction quality (atom-type / position / synthesis match rates).
Configuration lives in [configs/eval_vae.yaml](configs/eval_vae.yaml); the
checkpoint to evaluate is typically overridden on the command line:

```bash
python -m synlad.eval_autoencoder ckpt_path=/path/to/vae.ckpt
```

## Sampling from the diffusion model

[src/synlad/sample_eval_diffusion.py](src/synlad/sample_eval_diffusion.py)
draws samples from a trained latent diffusion checkpoint and writes the
decoded molecules (3D and synthesis pathways) to
`${DIR_NAME}/eval_outputs/`. Two configs are provided:

```bash
# Unconditional sampling
python -m synlad.sample_eval_diffusion --config-name=sample_uncond \
    DIR_NAME=/path/to/checkpoint_dir

# Pharmacophore-conditional sampling
python -m synlad.sample_eval_diffusion --config-name=sample_ph4_cond \
    DIR_NAME=/path/to/checkpoint_dir
```

Both configs default to `ckpt_path=${DIR_NAME}/last.ckpt`; override
`ckpt_path=...` to pick a specific checkpoint. Sampling requires the
reaction-predictor server to be running — see [Inference](#inference) below
for setup.

## Inference

At inference time the synthesis decoder calls out to an external **reaction
predictor** to expand each generated synthesis step. SynLaD talks to it over
HTTP, so you need to start the predictor as a Ray Serve deployment *before*
running `sample_eval_diffusion` (or any other inference entry point).

We use the predictor from [`rxn-lm`](https://github.com/john-bradshaw/rxn-lm),
pinned to commit `76ffe65` with the compatibility changes in
[`patches/rxn-lm-76ffe65.patch`](patches/rxn-lm-key-changes.patch). To bring it up:

1. Clone `rxn-lm` and apply the patch:

   ```bash
   git clone https://github.com/john-bradshaw/rxn-lm.git
   cd rxn-lm
   git checkout 76ffe65
   git apply /path/to/synlad/patches/rxn-lm-76ffe65.patch
   ```

2. Install `rxn-lm` into its own environment, following that repository's
   README.
3. From the `rxn-lm` checkout, launch the Ray Serve deployment that hosts the
   reaction predictor:

   ```bash
   cd scripts/serving
   WEIGHTS_PATH=<path to model weights> serve run serve:main_deployment
   ```

   This starts a local Ray instance and exposes the predictor HTTP endpoint
   that SynLaD will hit during sampling. Leave it running for the duration of
   your inference job and point SynLaD's reaction-predictor server URL at it.

## Evaluating per-pharmacophore hits

[scripts/evaluate_pharmacophore_hits.py](scripts/evaluate_pharmacophore_hits.py)
scores generated molecules against each pharmacophore's ground-truth reference
ligand using ROCS shape/colour overlap, then reports per-target hit counts,
unique scaffold counts, validity, uniqueness, and (optionally) diversity.
Aggregate metrics are written as JSON and per-pharmacophore counts as CSV in
the samples directory.

It expects `--samples_dir` to be a directory of the form produced by
`sample_eval_diffusion`, containing:

- `gt/molecule_ph_<i>.sdf` — reference ligand per pharmacophore `i`
- `pred/*_ph_<i>.sdf` — 3D-decoder predictions per pharmacophore (synlad)
- `synthesis_pathways_for_eval.csv` — synthesis-decoder products with a
  `pharmacophore_idx` column (synlad)

By default it runs evaluation for both the 3D and synthesis predictions of **synlad**:

```bash
python scripts/evaluate_pharmacophore_hits.py \
    --samples_dir /path/to/sampling/outputs \
    -n 50 -ns 100 --do_diversity
```

Other methods score a single SMILES dataframe (`--df_path`) or a fixed
molecule pool (`dataset_baseline`):

```bash
# Baselines that save generated SMILES per pharmacophore
python scripts/evaluate_pharmacophore_hits.py \
    --samples_dir /path/to/sampling/outputs \
    --method synformer --df_path /path/to/synformer_df.csv

# Dataset baseline: score a fixed pool of molecules against every pharmacophore
python scripts/evaluate_pharmacophore_hits.py \
    --samples_dir /path/to/sampling/outputs \
    --method dataset_baseline \
    --dataset_mols_path /path/to/smiles.txt --max_molecules 500
```

## Evaluating generated molecules

[scripts/evaluate_samples.py](scripts/evaluate_samples.py) computes general
quality metrics for a set of generated SMILES — validity, uniqueness, novelty
against a training set, and (optionally) FCD and AiZynthFinder
retrosynthesis.

It can optionally run
[AiZynthFinder](https://molecularai.github.io/aizynthfinder/) to assess
retrosynthetic accessibility (`--aizynth`). When enabled, you must point it at
an AiZynthFinder YAML config via `--aizynth_config`, listing the expansion
policy, filter policy, and stock files to use:

```bash
python scripts/evaluate_samples.py --input mols.smi --aizynth \
    --aizynth_config /path/to/aizynthfinder_config.yml
```

A template config is provided at
[examples/aizynthfinder_config.yml](examples/aizynthfinder_config.yml) — copy
it and replace the placeholder paths with the locations of your downloaded
models, templates, and stock files. See the
[AiZynthFinder configuration docs](https://molecularai.github.io/aizynthfinder/configuration.html)
for the full schema.

For exact-match novelty (`--training_smi`/`--training_csv`) and FCD
(`--calc_fcd`), the script needs the training-set SMILES. Place them under
[data/](data/) — by default we use [data/uspto_train_smiles.smi](data/uspto_train_smiles.smi),
one SMILES per line. Then point the evaluator at the file:

```bash
python scripts/evaluate_samples.py --input mols.smi \
    --training_smi data/uspto_train_smiles.smi \
    --calc_fcd
```

## Acknowledgements

`synlad` builds upon the source code of [all-atom-diffusion-transformer](https://github.com/facebookresearch/all-atom-diffusion-transformer/tree/main). Specifically, the 3D autoencoder and diffusion module largely follow its implementation.

## License

Code that is adapted from the all-atom-diffusion-transformer project (ADiT) is released under its original license (this is also marked in code headers). Our additional code (e.g., for synthesis component) is released under an MIT License. See [LICENSE](LICENSE) file for details.

## Citations

```
@inproceedings{
cretu2026synlad,
title={SynLaD: Latent Diffusion for Generating Synthesizable Molecules Conditioned on 3D Pharmacophore Profiles},
author={Miruna Cretu and John Bradshaw and Patricia Suriana and Saeed Saremi and Omar Mahmood and Kirill Shmilovich and Kangway V. Chuang and Vishnu Sresht and Colin A Grambow},
booktitle={Forty-third International Conference on Machine Learning},
year={2026},
url={https://openreview.net/forum?id=xn9Jxl54r3}
}
```
