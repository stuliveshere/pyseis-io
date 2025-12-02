import sys
from pathlib import Path

# Add src to path so pyseis_io can be imported
# This is critical for tests to run against the local source code
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
