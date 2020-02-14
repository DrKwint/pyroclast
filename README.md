# Pyroclast

by Eleanor Quint

A deep learning library by and for the Nebraska AI Research (NEAR) Lab

## Setup

We recommend running in a virtual environment with Python 3 and in a Unix-like OS.

In the project root, run

```python
virtualenv env -P python3
source env/bin/activate
```

That second command is used to activate the environment. This should be done every time before you try to run or install anything.

Then, to install the libraries this package depends on, pick a requirements.txt file (there may be multiple) and run:

```python
pip install -r <filename>.txt
```

Experiments can generally be run with:
```python -m pyroclast.eager_run --alg <module> ...```
Where the ellipsis could either be omitted or replaced with other arguments.

## Testing

This library uses PyTest. To run all tests:

```pytest pyroclast```

or, to run tests only for a particular module
```pytest pyroclast/<module>```

## Documentation

To build documentation on a Unix-like environment:

1. Make sure you've installed the `requirements.txt` file, as described above.

2. Navigate to the `docs/` directory. (Optional)

3. Run `./make_docs.sh`

## Architecture

Pyroclast's layout is patterned after OpenAI Baselines. There is a main run file, which dynamically loads a module and executes its `learn` function which is assumed to be located at `pyroclast.<module>.<module>.learn`. Hyperparameter defaults are located in `pyroclast.<module>.defaults` and are specified per dataset.

The `pyroclast.common` module contains code which is imported by any of the other modules. `cmd_util` specifies the top-level command line interface.
