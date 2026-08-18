"""Verify cudnn_path handles the namespace-package case that broke CI."""
import importlib.util
import os
import sys
import tempfile
import types

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location(
    'cudnn_path', os.path.join(_REPO, 'runpod', 'cudnn_path.py'))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

work = tempfile.mkdtemp()
libdir = os.path.join(work, 'lib')
os.makedirs(libdir)

parent = types.ModuleType('nvidia')
sys.modules['nvidia'] = parent


def install(module):
    """`import nvidia.cudnn` needs both sys.modules and the parent attribute."""
    sys.modules['nvidia.cudnn'] = module
    parent.cudnn = module


results = []

# 1. Namespace package: __file__ is None, __path__ set. Exactly what CI hit.
ns = types.ModuleType('nvidia.cudnn')
ns.__file__ = None
ns.__path__ = [work]
install(ns)
try:
    got = mod.cudnn_lib_dir()
    results.append(('namespace package, __file__ is None', got == libdir, got))
except Exception as e:
    results.append(('namespace package, __file__ is None', False, repr(e)))

# 2. Regular package: __file__ set, no __path__.
reg = types.ModuleType('nvidia.cudnn')
reg.__file__ = os.path.join(work, '__init__.py')
if hasattr(reg, '__path__'):
    del reg.__path__
install(reg)
try:
    got = mod.cudnn_lib_dir()
    results.append(('regular package, __file__ set', got == libdir, got))
except Exception as e:
    results.append(('regular package, __file__ set', False, repr(e)))

# 3. Neither: a clear error, not the TypeError that broke the build.
bad = types.ModuleType('nvidia.cudnn')
bad.__file__ = None
install(bad)
try:
    mod.cudnn_lib_dir()
    results.append(('no path info raises RuntimeError', False, 'did not raise'))
except RuntimeError as e:
    results.append(('no path info raises RuntimeError', True, str(e)))
except Exception as e:
    results.append(('no path info raises RuntimeError', False, repr(e)))

# 4. lib dir absent: reported, not silently wrong.
empty = tempfile.mkdtemp()
ns2 = types.ModuleType('nvidia.cudnn')
ns2.__file__ = None
ns2.__path__ = [empty]
install(ns2)
try:
    mod.cudnn_lib_dir()
    results.append(('missing lib dir raises', False, 'did not raise'))
except RuntimeError as e:
    results.append(('missing lib dir raises', True, str(e)[:44]))

fails = 0
for label, passed, detail in results:
    print('  [{}] {} - {}'.format('PASS' if passed else 'FAIL', label, detail))
    fails += 0 if passed else 1
print('{} passed, {} failed'.format(len(results) - fails, fails))


def test_everything_passed() -> None:
    """Surface the checks above to pytest as one assertion."""
    assert not fails, '{} checks failed'.format(fails)


if __name__ == '__main__':
    sys.exit(1 if fails else 0)
