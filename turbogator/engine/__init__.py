# from .reference_gatr import ReferenceGATrModel as TurboGatorModel
# from .baseline_gatr import BaselineGATrModel as TurboGatorModel
# define the active backend, comment the other one out pls

# from .scalar_opt1_gatr import ScalarOpt1GATrModel as TurboGatorModel
# from .optimised_gatr import OptimisedGATrModel as TurboGatorModel
from .vectorised_gatr import VectorisedGATrModel as TurboGatorModel

__all__ = ["TurboGatorModel"]
