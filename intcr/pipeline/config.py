EXPERIMENT_ROOT_KEY = 'root'


def simple_key_check(config, key):
    if key not in config:
        raise KeyError('The key "{}" should be present'.format(key))
