# define the active backend, comment the other one out pls
from .baseline_gatr import MVOnlyGATrModel as TurboGatorModel

__all__ = ["TurboGatorModel"]
