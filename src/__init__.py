"""Package initialisation for the sustainable_hr_tool project.

This module purposefully avoids importing heavy external libraries at
package import time to keep importing lightweight. Use explicit imports
inside modules to access pandas/numpy/matplotlib when required.
"""

# Export internal submodules for convenience
from . import (
    utils,
    data_loader,
    preprocessing,
    modelling,
    evaluation,
    evaluation_visuals,
    visualise,
    features,
    targets,
    train,
)

__all__ = [
    "utils",
    "data_loader",
    "preprocessing",
    "modelling",
    "evaluation",
    "evaluation_visuals",
    "visualise",
    "features",
    "targets",
    "train",
]
