mapping = {}


def register(name):

    def _thunk(func):
        mapping[name] = func
        return func

    return _thunk


# Implement datasets here


def get_dataset_builder(name):
    """
    Args:
        name (str): name under which a dataset has been registered

    Returns:
        data_dict_fn (dict): a function which takes kwargs and returns a data dictionary

    Notes:
        If you want to register your own dataset outside datasets.py, you can importy
        the `register` decorator and use it identically to how it's used in this file.
        Then, you can get the dataset from anywhere by importing this function.

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
    if name in mapping:
        return mapping[name]
    else:
        raise ValueError('Unknown dataset: {}'.format(name))


def check_datasets(name):
    return name in mapping.keys()
