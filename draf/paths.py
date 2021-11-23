from pathlib import Path

from appdirs import user_cache_dir


def _get_cache_directory() -> Path:
    """Returns the path to cache directory and creates it, if not yet existing."""
    fp = Path(user_cache_dir(appname="draf", appauthor="DrafProject"))
    fp.mkdir(parents=True, exist_ok=True)
    return fp


CACHE_DIR = _get_cache_directory()
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
