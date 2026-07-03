# Contributing to synlad

Thanks for your interest in contributing. This document covers the basics of
setting up a development environment, the project's coding conventions, and
how to submit changes.

## Development setup

The project is managed with [uv](https://docs.astral.sh/uv/). After cloning
the repository:

```bash
uv sync --extra dev                                  # core deps + editable install + dev tooling
uv sync --extra dev --extra openeye # also install OpenEye toolkits (requires license)
```

See the main [README](README.md) for platform requirements (Linux x86_64),
notes on the OpenEye license, and runtime configuration via Hydra.

## Tooling

We use [ruff](https://docs.astral.sh/ruff/) for linting and formatting, and
[pre-commit](https://pre-commit.com/) to run it on every commit. Once dev
deps are installed, hook up pre-commit:

```bash
uv run pre-commit install
```

You can also run the checks manually:

```bash
uv run ruff check src/ tests/
uv run ruff format src/ tests/
uv run pytest tests/
```

The ruff configuration lives in [pyproject.toml](pyproject.toml) under
`[tool.ruff]`. Line-length enforcement (E501) is currently disabled because
much of the existing code predates the lint config; please don't introduce
new lines longer than 100 characters in new code.

## Submitting changes

1. Fork the repository and create a feature branch off `main`.
2. Keep changes focused — one logical change per pull request.
3. Add or update tests when changing behaviour. Smoke tests live under
   `tests/`; some tests require real pre-processed pathway data and are
   gated on the `TEST_PATHWAY_DATA` environment variable (see the README).
4. Run `ruff check`, `ruff format`, and `pytest` before pushing.
5. Open a pull request with a clear description of the motivation and the
   change. Reference any related issues.

By contributing, you agree that your contributions will be licensed under
the MIT License (see [LICENSE](LICENSE)).
