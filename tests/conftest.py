import builtins
import sys
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")

# kaggle/__init__.py calls api.authenticate() on import and exits if no
# credentials are configured.  Pre-import it with exit() suppressed so that
# tests which mock KaggleApi can safely enter `with patch(...)` blocks without
# the first-time import triggering SystemExit.
if "kaggle" not in sys.modules:
    try:
        with patch.object(builtins, "exit"):
            import kaggle  # noqa: F401
    except Exception:
        pass
