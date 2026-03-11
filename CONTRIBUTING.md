## Pull Requests

### Architecture

Phantom uses a **service-oriented, event-driven architecture**. See [CLAUDE.md](CLAUDE.md) for current architecture guidelines.

**Key principles:**
- Services encapsulate one responsibility (FaceDetector, FaceSwapper, etc.)
- Use `CONFIG.set()` instead of global mutable state
- Emit events via `BUS.emit()` for inter-module communication
- Frame processors inherit from `FrameProcessor` ABC and chain without side effects

### Do

- ✓ Fix bugs before adding features
- ✓ One PR per feature or bug fix
- ✓ Consult on implementation details before starting
- ✓ Write complete type annotations (mypy strict)
- ✓ Run linting and tests before submitting
- ✓ Resolve CI pipeline failures
- ✓ Use clear naming; skip redundant comments
- ✓ Follow service-oriented patterns for new features

### Don't

- ✗ Introduce global mutable state
- ✗ Make direct calls between services (use EventBus)
- ✗ Ignore type requirements or use `# type: ignore`
- ✗ Submit massive code changes without discussion
- ✗ Submit proof-of-concepts or untested code
- ✗ Ignore code review feedback

## Code Style

See [CLAUDE.md](CLAUDE.md) for:
- Type checking (`mypy` strict mode)
- Linting (`flake8`)
- Naming conventions
- Documentation standards
