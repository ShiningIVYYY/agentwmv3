"""Resource ownership for WM runners; never changes baseline processes."""
import atexit
import os
from pathlib import Path
import tempfile


class PddlLibraryPool:
    """Keep native code mapped while reusing cleaned, exclusively owned engines.

    Early dlclose is unsafe for this DSO's native callbacks. A live environment
    and one serial Oracle branch need at most two libraries per worker.
    """
    def __init__(self, load, capacity=2):
        self.load = load
        self.capacity = capacity
        self.loaded = []
        self.idle = []
        self.active = {}

    def acquire(self):
        if self.idle:
            library = self.idle.pop()
        elif len(self.loaded) < self.capacity:
            library = self.load()
            self.loaded.append(library)
        else:
            raise RuntimeError('PDDL library pool exhausted: unexpected concurrent environment')
        self.active[id(library)] = library
        return library

    def release(self, library):
        if self.active.get(id(library)) is not library:
            raise RuntimeError('PDDL library released without exclusive ownership')
        library.cleanup()
        del self.active[id(library)]
        self.idle.append(library)


def configure_temporary_directory(output):
    parent = Path(output) / 'temporary'
    parent.mkdir(parents=True, exist_ok=True)
    owned = tempfile.TemporaryDirectory(prefix='runner-', dir=str(parent))
    # fast_downward copies its shared library via Python tempfile. Keep it on
    # the experiment volume rather than the shared root /tmp filesystem.
    os.environ['TMPDIR'] = owned.name
    tempfile.tempdir = owned.name
    atexit.register(owned.cleanup)
    return owned


def install_pddl_cleanup():
    import fast_downward
    import textworld
    from textworld.envs.pddl.pddl import PddlEnv
    if getattr(PddlEnv.close, '_agentwm_cleanup', False):
        return
    if PddlEnv.close is not textworld.Environment.close:
        raise RuntimeError('Unexpected TextWorld PDDL close implementation; review native cleanup')
    original = PddlEnv.close
    pool = PddlLibraryPool(fast_downward.load_lib)
    fast_downward.load_lib = pool.acquire
    def close(self):
        original(self)
        library = getattr(self, 'downward_lib', None)
        if library is not None:
            pool.release(library)
            self.downward_lib = None
    close._agentwm_cleanup = True
    close._agentwm_pool = pool
    PddlEnv.close = close
