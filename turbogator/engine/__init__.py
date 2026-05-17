# define the active backend, comment the other one out pls
# from .baseline_gatr import BaselineGATrModel as TurboGatorModel
from .optimised_gatr import OptimisedGATrModel as TurboGatorModel
# from .reference_gatr import ReferenceGATrModel as TurboGatorModel

__all__ = ["TurboGatorModel"]
