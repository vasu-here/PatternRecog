"""Run the project from a checkout without needing an editable installation."""

from pathlib import Path
import sys


# The package uses a src/ layout. Add that directory so this launcher works
# directly from the project root after installing requirements.txt.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from patternrecog.cli import main


if __name__ == "__main__":
    main()
