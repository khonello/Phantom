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

# 5. Two roots, the soname in the second. The pod case: a venv built with
#    --system-site-packages sees its own site-packages and the image's
#    dist-packages, and only one of them holds the cuDNN 9 that was installed.
sys_root = tempfile.mkdtemp()      # image's cuDNN 8 — listed first, wrong
venv_root = tempfile.mkdtemp()     # venv's cuDNN 9 — listed second, right
os.makedirs(os.path.join(sys_root, 'lib'))
os.makedirs(os.path.join(venv_root, 'lib'))
open(os.path.join(sys_root, 'lib', 'libcudnn.so.8'), 'w').close()
open(os.path.join(venv_root, 'lib', 'libcudnn.so.9'), 'w').close()
multi = types.ModuleType('nvidia.cudnn')
multi.__file__ = None
multi.__path__ = [sys_root, venv_root]
install(multi)
try:
    got = mod.cudnn_lib_dir()
    want = os.path.join(venv_root, 'lib')
    results.append(('picks the root holding libcudnn.so.9, not the first',
                    got == want, got))
except Exception as e:
    results.append(('picks the root holding libcudnn.so.9, not the first',
                    False, repr(e)))

# 6. Two roots, neither advertising the soname: still the first lib dir, since
#    the caller's dlopen is the real verdict and a rename is not a build break.
a, b = tempfile.mkdtemp(), tempfile.mkdtemp()
os.makedirs(os.path.join(a, 'lib'))
os.makedirs(os.path.join(b, 'lib'))
neither = types.ModuleType('nvidia.cudnn')
neither.__file__ = None
neither.__path__ = [a, b]
install(neither)
try:
    got = mod.cudnn_lib_dir()
    results.append(('no soname anywhere falls back to the first lib dir',
                    got == os.path.join(a, 'lib'), got))
except Exception as e:
    results.append(('no soname anywhere falls back to the first lib dir',
                    False, repr(e)))

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
