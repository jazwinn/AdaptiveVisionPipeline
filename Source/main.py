"""Launch the Adaptive Vision Pipeline GUI."""
import sys
from pathlib import Path

# Allow: python Source/main.py (run from the project root)
sys.path.insert(0, str(Path(__file__).parent.parent))

from Source.gui.app import main

if __name__ == "__main__":
    main()
