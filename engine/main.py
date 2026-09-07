import os
import sys

# PyInstaller with console=False sets sys.stdout/stderr to None.
# Redirect them immediately — before any other import — so that
# faulthandler, print(), and traceback all have somewhere to write.
_log_path = os.path.join(os.path.dirname(sys.executable)
                         if getattr(sys, "frozen", False)
                         else os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "main_error.txt")
if sys.stdout is None:
    sys.stdout = open(os.devnull, "w")
if sys.stderr is None:
    sys.stderr = open(_log_path, "w", buffering=1)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import traceback
import faulthandler
from engine.app import App

if __name__ == "__main__":
    faulthandler.enable()
    try:
        app = App()
        app.run()
    except BaseException as e:
        print("CRASH DETECTED IN MAIN:", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        # Also write to file explicitly in case stderr was already a terminal
        with open(_log_path, "a") as f:
            traceback.print_exc(file=f)
        sys.exit(1)