"""Package initialisation for the sustainable_hr_tool project.

Heavy modules (train, modelling) are intentionally excluded here to keep
package import lightweight. Import them explicitly where needed.
"""

from . import (
    utils,
    data_loader,
    preprocessing,
    evaluation,
    evaluation_visuals,
    visualise,
    features,
    targets,
    validation,
    race_predictor,
)

__all__ = [
    "utils",
    "data_loader",
    "preprocessing",
    "evaluation",
    "evaluation_visuals",
    "visualise",
    "features",
    "targets",
    "validation",
    "race_predictor",
]
