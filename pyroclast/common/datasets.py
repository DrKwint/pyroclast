mapping = {}


def register(name):

    def _thunk(func):
        mapping[name] = func
        return func

    return _thunk


# Implement datasets here


def get_dataset_builder(name):
    """
    If you want to register your own dataset outside datasets.py, you just need:

    Usage Example:
    -------------
    from pyroclast.common.datasets import register
    @register("your_dataset_name")
    def your_dataset_define(**ds_kwargs):
        ...
        return dataset_fn

    """
    if callable(name):
        return name
    elif name in mapping:
        return mapping[name]
    else:
        raise ValueError('Unknown dataset: {}'.format(name))


def check_datasets(name):
    return name in mapping.keys()
