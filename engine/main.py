import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import traceback
import faulthandler
from app import App

if __name__ == "__main__":
    faulthandler.enable()
    try:
        app = App()
        app.run()
    except BaseException as e:
        print("CRASH DETECTED IN MAIN:", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        with open("main_error.txt", "w") as f:
            traceback.print_exc(file=f)
        sys.exit(1)