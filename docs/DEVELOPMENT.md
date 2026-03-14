# Development Guide

This guide covers setting up your development environment, running tests, and contributing to Phantom.

## Development Setup

### 1. Clone & Environment

```bash
git clone <repo-url>
cd Phantom

# Create virtual environment
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

### 2. Install Dependencies

```bash
# Install base dependencies (CPU)
pip install -r requirements-pipeline-cpu.txt
# Or for GPU (CUDA):
pip install -r requirements-pipeline-gpu.txt

# For development (adds mypy, flake8, pytest)
pip install -r requirements-ci.txt

# For GPU support (optional)
# CUDA:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install onnxruntime-gpu

# OR ROCm (AMD):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.7
pip install onnxruntime-training
```

### 3. Verify Installation

Models are auto-downloaded on first run. Verify the environment:

```bash
# Test imports
python -c "from pipeline.config import CONFIG; from pipeline.events import BUS; print('OK')"

# Run full test with example files
python pipeline.py -s .github/examples/source.jpg -t .github/examples/target.mp4 -o /tmp/test.mp4
```

## Code Quality

### Type Checking (mypy)

Strict type checking is enforced. Run before committing:

```bash
mypy pipeline.py pipeline desktop
```

All functions and methods must have complete type annotations. No `# type: ignore` without justification.

### Linting with flake8

Check code style (E3, E4, F rules):

```bash
flake8 pipeline.py pipeline desktop
```

**Key Rules:**
- E3xx: Blank lines
- E4xx: Imports
- F: Undefined names, duplicate imports

**Configuration:** See `.flake8` for ignored checks.

### Type Checking with mypy

Verify type annotations (strict mode):

```bash
# Pipeline code
mypy pipeline.py roop

# Desktop controller code
mypy desktop.py desktop
```

**Configuration:** See `mypy.ini` for settings.

**Key Requirements:**
- All functions must have complete type annotations
- No `Any` types without explicit `# type: ignore` comments
- Return types always required

**Example:**
```python
def process_frame(frame: np.ndarray, face: Any) -> np.ndarray:
    """Process a frame with detected face."""
    return frame
```

### Running All Checks

Use the provided helper script:

**Linux/macOS:**
```bash
scripts/local-run-tests.sh
```

**Windows:**
```bash
scripts/local-run-tests.bat
```

Or manually:
```bash
flake8 pipeline.py roop && \
mypy pipeline.py roop && \
python pipeline.py -s=.github/examples/source.jpg -t=.github/examples/target.mp4 -o=test_output.mp4
```

## Code Style & Standards

### Functional Programming Only

**No OOP** — Classes only for framework requirements (CustomTkinter):

❌ Bad:
```python
class VideoProcessor:
    def __init__(self, path):
        self.path = path
    def process(self):
        pass
```

✅ Good:
```python
def process_video(path: str) -> None:
    pass
```

### Type Annotations

All functions must have complete types:

❌ Bad:
```python
def swap_face(frame, face):
    return frame
```

✅ Good:
```python
def swap_face(frame: np.ndarray, face: Any) -> np.ndarray:
    return frame
```

### Naming Conventions

Use self-documenting names instead of comments:

❌ Bad:
```python
def proc(f):
    # extract frame data
    d = f.shape
    return d
```

✅ Good:
```python
def extract_frame_dimensions(frame: np.ndarray) -> tuple[int, int, int]:
    return frame.shape
```

### Comments

Only comment non-obvious logic:

❌ Bad:
```python
x = 5  # set x to 5
for i in range(x):  # loop from 0 to 4
    print(i)
```

✅ Good:
```python
# ONNX Runtime requires batch dimension even for single image
batch_frame = np.expand_dims(frame, axis=0)
```

## Adding a Frame Processor

Frame processors are modular AI filters. Example: `face_swapper`, `face_enhancer`.

### 1. Create Module

Create `roop/processors/frame/my_processor.py`:

```python
import numpy as np
from typing import Any

# Global state for thread-safe model access
processor_model: Any = None


def pre_check() -> bool:
    """Verify processor dependencies available."""
    try:
        import torch
        return True
    except ImportError:
        return False


def pre_start() -> None:
    """Initialize models on startup."""
    global processor_model
    # Load your model here
    processor_model = load_model()


def post_end() -> None:
    """Cleanup after processing (optional)."""
    global processor_model
    processor_model = None


def process_frame(
    source_face: Any,
    target_face: Any,
    frame: np.ndarray,
) -> np.ndarray:
    """Process single frame.

    Args:
        source_face: Embedding of source face
        target_face: Detected target face data
        frame: Input frame (HxWx3 BGR uint8)

    Returns:
        Processed frame (HxWx3 BGR uint8)
    """
    # Your processing logic
    return frame
```

### 2. Register in CLI

Add to `roop/core.py` argument parser:

```python
parser.add_argument(
    '--frame-processor',
    nargs='+',
    default=['face_swapper'],
    choices=['face_swapper', 'face_enhancer', 'my_processor'],
    help='frame processors'
)
```

### 3. Test

```bash
python pipeline.py -s face.jpg -t video.mp4 -o output.mp4 --frame-processor my_processor
```

**See:** [`ARCHITECTURE.md`](ARCHITECTURE.md) for full processor interface details.

## Testing

### Unit Testing

Currently no unit tests (see CLAUDE.md). Tests would ideally cover:

- Face detection edge cases (no face, multiple faces)
- Frame processing pipeline
- Model loading and caching
- CLI argument parsing

### Integration Testing

Manual end-to-end test (in CI pipeline):

```bash
python pipeline.py \
  -s=.github/examples/source.jpg \
  -t=.github/examples/target.mp4 \
  -o=.github/examples/output.mp4
```

Verifies:
1. Argument parsing works
2. Model downloads/caches correctly
3. Face detection works
4. Frame processing pipeline succeeds
5. Output video created with correct audio/FPS

### Profiling Performance

To identify bottlenecks:

```python
import cProfile
import pstats

# In roop/predicter.py, wrap predict_video():
profiler = cProfile.Profile()
profiler.enable()
# ... processing code ...
profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative').print_stats(20)
```

## Debugging

### Enable Debug Output

Add debug logging to `roop/core.py`:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# In functions:
logger.debug(f"Processing frame: {i}")
```

### Inspect Temp Frames

Keep temporary frames for inspection:

```bash
python pipeline.py -s face.jpg -t video.mp4 -o output.mp4 --keep-frames
# Check: temp/video/0001.png, 0002.png, etc.
```

### Test on Small Data

Create short test video:

```bash
ffmpeg -i video.mp4 -ss 0 -t 5 -c copy test_5sec.mp4
python pipeline.py -s face.jpg -t test_5sec.mp4 -o test_output.mp4
```

## Contributing

### Before Submitting a PR

1. **Lint:** `flake8 pipeline.py roop desktop desktop.py` (must pass)
2. **Type check:** `mypy pipeline.py roop` (must pass; desktop/ checked separately)
3. **Test:** `python pipeline.py -s .github/examples/source.jpg -t .github/examples/target.mp4 -o test.mp4` (must complete)
4. **Review:** Read [CONTRIBUTING.md](../CONTRIBUTING.md) guidelines

### PR Guidelines

From CONTRIBUTING.md:

**Do:**
- Fix bugs over adding features
- One PR per feature/fix
- Consult before major changes
- Test before submission
- Resolve CI failures

**Don't:**
- Introduce OOP or architectural changes
- Ignore requirements
- Submit massive code changes
- Submit POCs or undocumented APIs
- Bypass code style/types

### PR Checklist

- [ ] Linting passes: `flake8 pipeline.py roop desktop desktop.py`
- [ ] Type checking passes: `mypy pipeline.py roop` (and `mypy desktop.py desktop` if desktop code changed)
- [ ] Integration test passes: Example video processes correctly
- [ ] No new OOP classes
- [ ] All functions have complete type annotations
- [ ] No new comments (use self-documenting names)
- [ ] Follows existing code patterns
- [ ] One feature/fix per PR

## CI Pipeline

The CI pipeline runs on every push/PR:

1. **Lint:** `flake8 pipeline.py roop`
2. **Type check:** `mypy pipeline.py roop`
3. **Integration test:** Process example video

**Configuration:** `.github/workflows/ci.yml`

Failing CI blocks merge. Fix failures locally and push again.

## Useful Commands

### Clean Up

```bash
# Remove temp directories
rm -rf temp/

# Remove test outputs
rm test_output.mp4

# Remove __pycache__
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null
```

### Quick Test Loop

```bash
# Run all checks
flake8 pipeline.py roop && mypy pipeline.py roop && echo "✓ All checks passed"
```

### Update Dependencies

```bash
# Pin versions
pip freeze > requirements-locked.txt

# Update a package
pip install --upgrade torch
```

## Two Entry Points

roop-cam has two entry points with separate concerns:

| Entry Point | Purpose | Dependencies |
|-------------|---------|--------------|
| `pipeline.py` | AI pipeline — GUI, CLI, headless live | `requirements-pipeline-cpu.txt` / `requirements-pipeline-gpu.txt` |
| `desktop.py` | Pipeline controller — interactive REPL, scripting | `requirements-desktop.txt` (stdlib-only now) |

When making changes that affect both (e.g. adding a new control command), update `roop/api/schema.py` first — it is the shared contract between the two processes.

## Resources

- [ARCHITECTURE.md](ARCHITECTURE.md) — System design, data flow, client-server model
- [REALTIME_PIPELINE.md](REALTIME_PIPELINE.md) — Live pipeline phases, CLI flags, production examples
- [USAGE.md](USAGE.md) — User-facing features, live mode, desktop controller
- [CONTRIBUTING.md](../CONTRIBUTING.md) — PR guidelines
- [mypy docs](https://mypy.readthedocs.io/) — Type checking
- [flake8 docs](https://flake8.pycqa.org/) — Linting

## Questions?

- Check [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
- Review existing code in `roop/` directory
- Search GitHub issues for similar questions
- Create new issue with detailed context
