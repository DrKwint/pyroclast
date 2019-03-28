import tensorflow_datasets as tfds


def mnist():
    return tfds.load(name="mnist")


datasets = {'mnist': mnist}
tasks = {'supervised_classification': 'mnist'}


def load_data(task_id, task_type, seed):
    if task_type == 'supervised_classification':
        return datasets[task_id]()
    else:
        assert False
