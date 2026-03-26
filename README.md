# Typed LISA Toolkit

[![pipeline status](https://gitlab.in2p3.fr/lisa-apc/typed-lisa-toolkit/badges/main/pipeline.svg)](https://gitlab.in2p3.fr/lisa-apc/typed-lisa-toolkit/-/commits/main)

Typed LISA Toolkit provides a concrete implementation for the data
analysis workflow components within CU L2D. It follows the API
contracts defined in `l2d-interface`.

- [Documentation](https://lisa-apc.pages.in2p3.fr/typed-lisa-toolkit/)
- [L2D Interface](https://l2d-interface-c43116.pages.in2p3.fr/)

## Installation

### As Dependency

Add "typed-lisa-toolkit @ git+https://gitlab.in2p3.fr/lisa-apc/typed-lisa-toolkit.git" in `dependencies` of `pyproject.toml`.

The project is planned to be released to PyPI once the version `1.0.0` is reached.

### Development

It is recommended to use `uv`. Clone the project and `cd` into its directory, then run `uv pip install -e .`.

## Static type checking

This project tries to privide type safe infrastructure for CU L2D. The project uses [basedpyright](https://docs.basedpyright.com/latest/) to check the type correctness.