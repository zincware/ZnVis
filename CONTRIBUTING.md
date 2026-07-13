# Contributing to ZnVis

Thanks for your interest in improving ZnVis! Contributions of all kinds are
welcome, from bug reports and documentation fixes to new mesh types and
rendering features.

## Reporting bugs and requesting features

Please open an issue on the
[GitHub issue tracker](https://github.com/zincware/ZnVis/issues). When reporting
a bug, include:

- A short description of what you expected to happen and what happened instead.
- A minimal code snippet that reproduces the problem.
- Your operating system, Python version, and ZnVis version
  (`python -c "import znvis; print(znvis.__version__)"`).

## Getting help

If you have a usage question rather than a bug report, please start a thread on
[GitHub Discussions](https://github.com/zincware/ZnVis/discussions) or open an
issue. The [documentation](https://znvis.readthedocs.io) also contains a user
guide with worked examples.

## Development setup

ZnVis uses [uv](https://docs.astral.sh/uv/) for environment management, though a
plain `pip` virtual environment works too.

```sh
git clone https://github.com/zincware/ZnVis.git
cd ZnVis
uv sync --extra dev          # or: pip install -e ".[dev]"
pre-commit install
```

## Running the tests

The test suite lives under `CI/` and is run with `pytest`:

```sh
uv run pytest .              # or: pytest .
```

Please make sure the full suite passes before opening a pull request, and add
tests that cover any new behaviour.

## Code style

Code is formatted with [black](https://github.com/psf/black) and
[isort](https://pycqa.github.io/isort/), and linted with
[flake8](https://flake8.pycqa.org/). These run automatically through
`pre-commit`; you can also run them manually:

```sh
pre-commit run --all-files
```

## Submitting changes

1. Fork the repository and create a feature branch.
2. Make your changes, keeping commits focused and adding tests where relevant.
3. Ensure the tests pass and the pre-commit hooks are clean.
4. Open a pull request against `main` describing the change and its motivation.

By contributing, you agree that your contributions will be licensed under the
project's [EPL-2.0 license](LICENSE).
