"""
Studio swap backends — whole pipelines run as separate processes.

Selected by `config.studio_swapper`, described in
`pipeline/services/studio_swappers.py`, and refused outright on the live path.

**Why a subprocess and not an import.** Every backend here ships as a repository
to clone rather than a package to install, and each vendors its own copy of a
diffusion stack: REFace carries an old latent-diffusion tree with
`taming-transformers` and its own CLIP, GHOST 2.0 carries StyleMatte and a LaMa
inpainter, DreamID-V carries a fork of Wan 2.1 and wants torch >= 2.4. Those
requirements conflict with each other and with this pipeline's. Importing any
one of them into this process would decide the torch version for all of them,
and importing two is not possible at all.

So each backend names its own interpreter and its own checkout, and the contract
between us is the filesystem: we write a source and a target where it expects
them, run its own documented entrypoint, and read back what it produced. That
also means their internal APIs can drift without breaking us — only the command
line is depended on, which is the part their own documentation pins.

**What is deliberately not done here.** No compositing. `FaceCompositor` matches
colour and detail to the target and masks with the target's landmark hull, and
every one of those steps would put back the target information these models
exist to remove — two of the three swap the whole head, and a hull mask would
clip that straight back to a face swap. The backend's output *is* the result.

**The rule a refusal must not break.** A photo that cannot be swapped writes no
file (see `guards.py`), because a copy of the input wearing the output's name is
indistinguishable from success to whoever opens the folder. A backend that
fails, times out, or produces nothing therefore returns False and leaves the
destination absent, rather than falling back to the input.
"""

import glob
import os
import shutil
import subprocess
import tempfile
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Type

from pipeline.config import FaceSwapConfig
from pipeline.io.ffmpeg import IMAGE_EXTENSIONS
from pipeline.logging import emit_error, emit_status, emit_warning
from pipeline.services import studio_swappers
from pipeline.services.studio_swappers import StudioSwapperModel

# How much longer than the model's own estimate to wait before giving up.
#
# Generous because these are diffusion models on shared hardware and a first run
# also pays weight loading, and bounded because the alternative is a hung
# subprocess holding a paid GPU with nobody watching. A timeout is reported as a
# failure, which under the rule above means no output file rather than a
# truncated one.
_TIMEOUT_FACTOR = 20.0
_TIMEOUT_FLOOR_SECONDS = 300.0


class StudioSwapper(ABC):
    """
    One studio backend, driven through its own documented command line.

    Subclasses supply the argv and say where the result lands; everything about
    running it — availability, the working directory, the timeout, collecting
    the output and cleaning up — is decided once here, for the same reason
    `onnx_session.py` owns session construction.
    """

    def __init__(self, model: StudioSwapperModel, config: FaceSwapConfig) -> None:
        """
        Args:
            model: Registry entry this backend runs
            config: Pipeline config, for the execution provider and log scope
        """
        self.model = model
        self.config = config

    # ------------------------------------------------------------------
    # Availability
    # ------------------------------------------------------------------

    def repo(self) -> str:
        """
        The configured checkout, or an empty string.

        Returns:
            Absolute path to the repository, expanded, or '' when unset
        """
        raw = os.environ.get(self.model.repo_env, '').strip()
        return os.path.abspath(os.path.expanduser(raw)) if raw else ''

    def interpreter(self) -> str:
        """
        The interpreter to run the checkout's entrypoint with.

        Deliberately has no fallback to `sys.executable`. This process's
        environment is the one that cannot satisfy these dependencies — that is
        the whole reason for a subprocess — so falling back to it would produce
        an ImportError from inside the child and present as "the model is
        broken" rather than "it was never configured".

        Returns:
            Absolute path to the interpreter, or '' when unset
        """
        raw = os.environ.get(self.model.python_env, '').strip()
        return os.path.abspath(os.path.expanduser(raw)) if raw else ''

    def available(self) -> Tuple[bool, str]:
        """
        Whether this backend can run, and why not when it cannot.

        Returns:
            (ready, reason). `reason` is empty when ready, and otherwise names
            the specific missing thing and the variable that supplies it
        """
        repo = self.repo()
        if not repo:
            return False, '{} is not set'.format(self.model.repo_env)
        if not os.path.isdir(repo):
            return False, '{}={} is not a directory'.format(
                self.model.repo_env, repo)

        entry = os.path.join(repo, self.model.entrypoint)
        if not os.path.isfile(entry):
            return False, '{} has no {} — is it the right checkout?'.format(
                repo, self.model.entrypoint)

        python = self.interpreter()
        if not python:
            return False, (
                '{} is not set. This backend needs its own virtualenv — ours '
                'cannot satisfy its requirements, which is the whole reason it '
                'runs as a subprocess.'.format(self.model.python_env))
        if not os.path.isfile(python):
            return False, '{}={} does not exist'.format(
                self.model.python_env, python)

        return self._extra_requirements(repo)

    def _extra_requirements(self, repo: str) -> Tuple[bool, str]:
        """
        Backend-specific checks beyond a checkout and an interpreter.

        Args:
            repo: The verified checkout

        Returns:
            (ready, reason), defaulting to ready
        """
        return True, ''

    # ------------------------------------------------------------------
    # Running
    # ------------------------------------------------------------------

    def swap(
        self,
        source_path: str,
        target_path: str,
        output_path: str,
    ) -> bool:
        """
        Swap one target, writing the result to `output_path`.

        Args:
            source_path: The operator's face
            target_path: The image or video to swap it into
            output_path: Where the finished result should land

        Returns:
            True only if `output_path` exists afterwards and came from the
            model. On any failure the destination is left absent, since an
            unswapped copy of the input is worse than no file at all
        """
        ready, reason = self.available()
        if not ready:
            emit_error(
                '{} is selected but unavailable: {}'.format(
                    self.model.name, reason),
                scope='STUDIO',
            )
            return False

        if not self._media_matches(target_path):
            return False

        work = tempfile.mkdtemp(prefix='phantom-studio-')
        try:
            argv = self._argv(source_path, target_path, output_path, work)
            emit_status(
                '{}: swapping {} ({}px, ~{:.0f}s expected)'.format(
                    self.model.name, os.path.basename(target_path),
                    self.model.resolution, self.model.seconds_per_item),
                scope='STUDIO',
            )
            if not self._run(argv, work):
                return False

            produced = self._collect(output_path, work)
            if not produced:
                emit_error(
                    '{} exited cleanly but produced no output. Nothing was '
                    'written for {}.'.format(
                        self.model.name, os.path.basename(target_path)),
                    scope='STUDIO',
                )
            return produced
        finally:
            shutil.rmtree(work, ignore_errors=True)

    def _media_matches(self, target_path: str) -> bool:
        """
        Whether this backend takes the kind of target it was handed.

        Refused rather than attempted: the two image backends given a clip
        would swap one frame, and the video backend given a still would fail
        deep inside its own dataloader with a message about tensors.

        Args:
            target_path: The target

        Returns:
            True when the media kind matches the registry entry
        """
        is_image = os.path.splitext(target_path)[1].lower() in IMAGE_EXTENSIONS
        wanted_image = self.model.media == 'image'
        if is_image == wanted_image:
            return True

        emit_error(
            '{} swaps {}s and was given {}.'.format(
                self.model.name, self.model.media,
                'an image' if is_image else 'a video'),
            scope='STUDIO',
        )
        return False

    def _run(self, argv: List[str], work: str) -> bool:
        """
        Run the backend's entrypoint from inside its own checkout.

        `cwd` is the checkout because every one of these resolves configs,
        vendored packages and default checkpoint paths relative to it — running
        from anywhere else fails on an import rather than on a flag.

        Args:
            argv: Full command line
            work: Scratch directory, exported so the child can use it

        Returns:
            True on a zero exit code within the timeout
        """
        timeout = max(
            _TIMEOUT_FLOOR_SECONDS,
            self.model.seconds_per_item * _TIMEOUT_FACTOR,
        )
        environment = dict(os.environ)
        environment['PYTHONPATH'] = self.repo()
        environment['PHANTOM_STUDIO_WORK'] = work

        try:
            completed = subprocess.run(
                argv,
                cwd=self.repo(),
                env=environment,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            emit_error(
                '{} exceeded {:.0f}s and was stopped. No output written.'
                .format(self.model.name, timeout),
                scope='STUDIO',
            )
            return False
        except OSError as e:
            emit_error(
                'Could not start {}: {}: {}'.format(
                    self.model.name, type(e).__name__, e),
                exception=e, scope='STUDIO',
            )
            return False

        if completed.returncode == 0:
            return True

        # The last few lines, not the whole log: these print per-step progress
        # and the useful part is always the traceback at the end.
        tail = '\n'.join(
            (completed.stderr or completed.stdout or '').strip().splitlines()[-8:])
        emit_error(
            '{} exited {}. Last output:\n{}'.format(
                self.model.name, completed.returncode, tail),
            scope='STUDIO',
        )
        return False

    # ------------------------------------------------------------------
    # Per-backend
    # ------------------------------------------------------------------

    @abstractmethod
    def _argv(
        self,
        source_path: str,
        target_path: str,
        output_path: str,
        work: str,
    ) -> List[str]:
        """
        The command line for one swap.

        Args:
            source_path: The operator's face
            target_path: What to swap it into
            output_path: Where the result should land, for backends that take
                         a destination directly
            work: Scratch directory for backends that take directories

        Returns:
            argv, beginning with the interpreter
        """

    @abstractmethod
    def _collect(self, output_path: str, work: str) -> bool:
        """
        Move whatever the backend produced to `output_path`.

        Args:
            output_path: Where the result belongs
            work: Scratch directory the run used

        Returns:
            True if `output_path` exists afterwards
        """

    @staticmethod
    def _newest(patterns: List[str]) -> Optional[str]:
        """
        The most recently modified file matching any of these globs.

        Backends that write into a results *directory* name their outputs by
        their own conventions — an index, a timestamp, a grid — and those
        conventions change between commits. Taking the newest file is stable
        against that in a way that reproducing their naming is not, and the
        directory is a fresh scratch one, so there is nothing older to confuse
        it with.

        Args:
            patterns: Glob patterns to search

        Returns:
            Path to the newest match, or None
        """
        found: List[str] = []
        for pattern in patterns:
            found.extend(p for p in glob.glob(pattern) if os.path.isfile(p))
        if not found:
            return None
        return max(found, key=os.path.getmtime)

    @staticmethod
    def _deliver(produced: Optional[str], output_path: str) -> bool:
        """
        Put a produced file at its destination.

        Args:
            produced: What the backend made, or None
            output_path: Where it belongs

        Returns:
            True if the destination exists afterwards
        """
        if not produced or not os.path.isfile(produced):
            return False

        parent = os.path.dirname(os.path.abspath(output_path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        shutil.move(produced, output_path)
        return os.path.isfile(output_path)


class RefaceSwapper(StudioSwapper):
    """
    REFace — diffusion inpainting, 512px, swaps the head rather than the face.

    Its entrypoint takes *folders* rather than files and writes into a results
    directory, so a single swap is staged as one-image folders in scratch. That
    is its documented interface (`Demo.sh`), and driving the documented one is
    what keeps this working across their commits.

    `--scale` and `--ddim_steps` are the published defaults. Steps is the lever
    worth touching first: the authors report usable output at 5 against the
    default 50, which is the difference between ~4.7s and under a second.
    """

    def _extra_requirements(self, repo: str) -> Tuple[bool, str]:
        """
        REFace needs a config and a checkpoint neither of which it bundles.

        Args:
            repo: The verified checkout

        Returns:
            (ready, reason)
        """
        for variable, default in (
            ('REFACE_CONFIG', 'models/REFace/configs/project_ffhq.yaml'),
            ('REFACE_CKPT', 'models/REFace/checkpoints/saved.ckpt'),
        ):
            path = self._resolve(repo, variable, default)
            if not os.path.isfile(path):
                return False, (
                    '{} not found at {} — set {} or place it there'.format(
                        os.path.basename(default), path, variable))
        return True, ''

    @staticmethod
    def _resolve(repo: str, variable: str, default: str) -> str:
        """
        A path from the environment, or the repository's own default location.

        Args:
            repo: Checkout root
            variable: Environment variable that may override it
            default: Repo-relative fallback, as the project's own docs give it

        Returns:
            Absolute path
        """
        raw = os.environ.get(variable, '').strip()
        if raw:
            return os.path.abspath(os.path.expanduser(raw))
        return os.path.join(repo, default)

    def _argv(
        self,
        source_path: str,
        target_path: str,
        output_path: str,
        work: str,
    ) -> List[str]:
        """See base class."""
        repo = self.repo()
        sources = os.path.join(work, 'source')
        targets = os.path.join(work, 'target')
        results = os.path.join(work, 'results')
        base = os.path.join(work, 'base')
        for directory in (sources, targets, results, base):
            os.makedirs(directory, exist_ok=True)

        # Copied rather than symlinked: the child runs as another process and
        # may run as another user on a pod, and a broken link presents as "no
        # face detected" rather than as a missing file.
        shutil.copy2(source_path, os.path.join(
            sources, os.path.basename(source_path)))
        shutil.copy2(target_path, os.path.join(
            targets, os.path.basename(target_path)))

        steps = os.environ.get('REFACE_DDIM_STEPS', '50').strip() or '50'
        scale = os.environ.get('REFACE_SCALE', '3.5').strip() or '3.5'

        return [
            self.interpreter(), self.model.entrypoint,
            '--outdir', results,
            '--target_folder', targets,
            '--src_folder', sources,
            '--config', self._resolve(
                repo, 'REFACE_CONFIG',
                'models/REFace/configs/project_ffhq.yaml'),
            '--ckpt', self._resolve(
                repo, 'REFACE_CKPT',
                'models/REFace/checkpoints/saved.ckpt'),
            '--Base_dir', base,
            '--n_samples', '1',
            '--scale', scale,
            '--ddim_steps', steps,
        ]

    def _collect(self, output_path: str, work: str) -> bool:
        """See base class."""
        results = os.path.join(work, 'results')
        patterns = [
            os.path.join(results, '**', '*' + extension)
            for extension in IMAGE_EXTENSIONS
        ]
        return self._deliver(self._newest(patterns), output_path)

    @staticmethod
    def _newest(patterns: List[str]) -> Optional[str]:
        """
        As the base class, but recursive — REFace nests its results.

        Args:
            patterns: Glob patterns, containing `**`

        Returns:
            Path to the newest match, or None
        """
        found: List[str] = []
        for pattern in patterns:
            found.extend(
                p for p in glob.glob(pattern, recursive=True)
                if os.path.isfile(p))
        if not found:
            return None
        return max(found, key=os.path.getmtime)


class Ghost2Swapper(StudioSwapper):
    """
    GHOST 2.0 — head transfer with explicit background inpainting.

    The only one of the three whose entrypoint takes a source, a target and a
    destination directly, so there is nothing to stage and nothing to search
    for afterwards.

    `--use_kandi` enables its post-blending pass, which runs a Kandinsky
    inpainting pipeline over the seam. Off by default here because it is a
    second diffusion model on top of an already slow path, and because the seam
    is the thing this codebase judges by eye — turning it on should be a
    deliberate A/B rather than a default nobody chose.
    """

    def _extra_requirements(self, repo: str) -> Tuple[bool, str]:
        """
        Both checkpoints must exist; the repository ships neither.

        Args:
            repo: The verified checkout

        Returns:
            (ready, reason)
        """
        for variable, default in (
            ('GHOST2_CKPT_ALIGNER',
             'aligner_checkpoints/aligner_1020_gaze_final.ckpt'),
            ('GHOST2_CKPT_BLENDER', 'blender_checkpoints/blender_lama.ckpt'),
        ):
            path = RefaceSwapper._resolve(repo, variable, default)
            if not os.path.isfile(path):
                return False, '{} not found at {} — set {}'.format(
                    os.path.basename(default), path, variable)
        return True, ''

    def _argv(
        self,
        source_path: str,
        target_path: str,
        output_path: str,
        work: str,
    ) -> List[str]:
        """See base class."""
        repo = self.repo()
        # Written into scratch and moved on success, so a failed run cannot
        # leave a partial file where the caller expects a finished one.
        staged = os.path.join(work, 'result.png')

        argv = [
            self.interpreter(), self.model.entrypoint,
            '--source', os.path.abspath(source_path),
            '--target', os.path.abspath(target_path),
            '--save_path', staged,
            '--config_a', RefaceSwapper._resolve(
                repo, 'GHOST2_CONFIG_ALIGNER', 'configs/aligner.yaml'),
            '--config_b', RefaceSwapper._resolve(
                repo, 'GHOST2_CONFIG_BLENDER', 'configs/blender.yaml'),
            '--ckpt_a', RefaceSwapper._resolve(
                repo, 'GHOST2_CKPT_ALIGNER',
                'aligner_checkpoints/aligner_1020_gaze_final.ckpt'),
            '--ckpt_b', RefaceSwapper._resolve(
                repo, 'GHOST2_CKPT_BLENDER',
                'blender_checkpoints/blender_lama.ckpt'),
        ]
        if os.environ.get('GHOST2_USE_KANDI', '').strip().lower() in (
            '1', 'true', 'yes',
        ):
            argv.append('--use_kandi')
        return argv

    def _collect(self, output_path: str, work: str) -> bool:
        """See base class."""
        return self._deliver(os.path.join(work, 'result.png'), output_path)


class DreamIdVSwapper(StudioSwapper):
    """
    DreamID-V — a diffusion transformer that swaps a *clip*, not frames.

    The only video-native backend here, which is the argument for it: swapping
    frames independently and smoothing afterwards is what the temporal EMA in
    `FaceCompositor` exists to paper over, and a model that attends across time
    does not need papering over.

    It needs two checkpoint trees rather than one — its own weights plus the
    Wan 2.1 VAE and text encoder — and 16 GB of VRAM. On a rented pod that is
    the same card the live pipeline uses, so a render here is exclusive of a
    call rather than concurrent with it.
    """

    def _extra_requirements(self, repo: str) -> Tuple[bool, str]:
        """
        The Wan backbone and this model's own checkpoint.

        Args:
            repo: The verified checkout

        Returns:
            (ready, reason)
        """
        wan = os.environ.get('DREAMIDV_WAN_DIR', '').strip()
        if not wan or not os.path.isdir(os.path.expanduser(wan)):
            return False, (
                'DREAMIDV_WAN_DIR must point at a Wan2.1-T2V-1.3B checkout '
                '(it supplies the VAE and text encoder)')

        checkpoint = os.environ.get('DREAMIDV_CKPT', '').strip()
        if not checkpoint or not os.path.isfile(os.path.expanduser(checkpoint)):
            return False, 'DREAMIDV_CKPT must point at dreamidv_faster.pth'

        return True, ''

    def _argv(
        self,
        source_path: str,
        target_path: str,
        output_path: str,
        work: str,
    ) -> List[str]:
        """See base class."""
        staged = os.path.join(work, 'result.mp4')
        size = os.environ.get('DREAMIDV_SIZE', '').strip() or '832*480'
        steps = os.environ.get('DREAMIDV_STEPS', '').strip() or '16'
        seed = os.environ.get('DREAMIDV_SEED', '').strip() or '42'

        return [
            self.interpreter(), self.model.entrypoint,
            '--size', size,
            '--ckpt_dir', os.path.abspath(os.path.expanduser(
                os.environ.get('DREAMIDV_WAN_DIR', ''))),
            '--dreamidv_ckpt', os.path.abspath(os.path.expanduser(
                os.environ.get('DREAMIDV_CKPT', ''))),
            '--ref_image', os.path.abspath(source_path),
            '--ref_video', os.path.abspath(target_path),
            '--save_file', staged,
            '--sample_steps', steps,
            '--base_seed', seed,
        ]

    def _collect(self, output_path: str, work: str) -> bool:
        """See base class."""
        produced = os.path.join(work, 'result.mp4')
        if os.path.isfile(produced):
            return self._deliver(produced, output_path)

        # It names its own file from the task and a timestamp when `--save_file`
        # is not honoured, and it writes into the working directory rather than
        # into scratch. Looked for rather than assumed, since a run that
        # succeeded and was then reported as producing nothing is the worst of
        # both outcomes on a path this expensive.
        return self._deliver(
            self._newest([os.path.join(self.repo(), '*.mp4')]), output_path)


_BACKENDS: Dict[str, Type[StudioSwapper]] = {
    'reface': RefaceSwapper,
    'ghost_2': Ghost2Swapper,
    'dreamid_v': DreamIdVSwapper,
}


def resolve_backend(config: FaceSwapConfig) -> Optional[StudioSwapper]:
    """
    The studio backend this config selects, if any.

    Args:
        config: Pipeline config

    Returns:
        A backend, or None when `studio_swapper` is unset. An unknown name is
        reported and treated as unset rather than raising, so a typo in `.env`
        degrades to the ordinary compositing path instead of failing a job
    """
    name = str(getattr(config, 'studio_swapper', '') or '').strip()
    if not name:
        return None

    try:
        model = studio_swappers.resolve(name)
    except KeyError as e:
        emit_warning(str(e), scope='STUDIO')
        return None

    return _BACKENDS[model.name](model, config)
