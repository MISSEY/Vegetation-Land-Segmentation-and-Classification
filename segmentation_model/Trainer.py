import detectron2.engine as engine
import weakref
class Basehook(engine.HookBase):
    def after_step(self):
        # if self.trainer.iter % 100 == 0:
        print(f"Hello at iteration {self.trainer.iter}!")

    def before_train(self):
        """
        Called before the first iteration.
        """
        pass

    def after_train(self):
        """
        Called after the last iteration.
        """
        pass

    def before_step(self):
        """
        Called before each iteration.
        """
        print(f"Hello at iteration {self.trainer.iter}!")

class Trainer(engine.TrainerBase):
    def __init__(self):
        super.__init__()

        self._hooks = []
    def register_hooks(self, hooks):
        """
        Register hooks to the trainer. The hooks are executed in the order
        they are registered.

        Args:
            hooks (list[Optional[HookBase]]): list of hooks
        """
        hooks = [h for h in hooks if h is not None]
        for h in hooks:
            assert isinstance(h, Basehook)
            # To avoid circular reference, hooks and trainer cannot own each other.
            # This normally does not matter, but will cause memory leak if the
            # involved objects contain __del__:
            # See http://engineering.hearsaysocial.com/2013/06/16/circular-references-in-python/
            h.trainer = weakref.proxy(self)
        self._hooks.extend(hooks)


