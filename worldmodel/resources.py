"""Resource ownership for WM runners; never changes baseline processes."""
import atexit
import os
from pathlib import Path
import tempfile


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
    def close(self):
        original(self)
        library = getattr(self, 'downward_lib', None)
        if library is not None:
            self.downward_lib = None
            fast_downward.close_lib(library)
    close._agentwm_cleanup = True
    PddlEnv.close = close
