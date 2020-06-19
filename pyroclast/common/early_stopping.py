import sys


class EarlyStopping(object):
    """Object which handles tracking early stopping and checkpointing when model improves"""

    def __init__(self,
                 patience,
                 ckpt_manager=None,
                 minimize_metric=True,
                 max_epochs=None):
        # internally, always minimize
        self._best_score = sys.float_info.max
        self._score_sign = 1. if minimize_metric else -1.

        self.counter = 0
        self.patience = patience
        self.max_epochs = max_epochs

        self.ckpt_manager = ckpt_manager

    def __call__(self, epoch, score):
        # Returns True when training should stop
        if self.max_epochs is not None and epoch >= self.max_epochs:
            return True
        if score * self._score_sign < self._best_score - (self._best_score /
                                                          200.):
            self._best_score = score * self._score_sign
            self.counter = 0
            if self.ckpt_manager is not None:
                try:
                    self.ckpt_manager.save(checkpoint_number=epoch)
                except Exception as e:
                    print(e)
                    print("CANNOT SAVE THE MODEL!")
            return False
        else:
            self.counter += 1
            return self.counter > self.patience
