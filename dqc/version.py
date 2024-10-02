import platform

import accelerate
import sentence_transformers
import sklearn
import transformers

__version__ = "0.2.0"


def show_versions():
    """Print useful debugging information

    Returns:
        dict: Dictionary object containing system information
    """
    versions = {
        "os_type": platform.system(),
        "os_version": platform.release(),
        "python_version": platform.python_version(),
        "dqc-toolkit": __version__,
        "transformers": transformers.__version__,
        "sentence_transformers": sentence_transformers.__version__,
        "accelerate": accelerate.__version__,
        "scikit-learn": sklearn.__version__,
    }

    return versions
