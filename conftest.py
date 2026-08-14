"""Put the repository root on sys.path for the whole test session.

`tests/layernorm/` is a package (it has an `__init__.py`), so under pytest's
default prepend import mode the inserted path is `tests/`, not the repository
root — and every `import kernels...` / `import benchmarks...` in those nine
modules raised ModuleNotFoundError at collection time. A conftest.py here makes
pytest insert the root instead, so `pytest tests/` collects cleanly.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
