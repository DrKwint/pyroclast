class dummy_context_mgr():

    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc_value, traceback):
        return False
