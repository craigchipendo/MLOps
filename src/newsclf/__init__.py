"""News topic classifier: an end-to-end, production-style ML lifecycle project."""

import os
import warnings

# Silence TensorFlow C++ start-up noise before TF is imported anywhere in the package.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
warnings.filterwarnings("ignore", message=".*`shuffle=True` was passed.*")

__version__ = "1.0.0"
