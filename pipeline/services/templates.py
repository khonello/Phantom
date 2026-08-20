"""
Bundled target templates — curated scenes the source face is swapped into.

A template is a **target photo that ships with the product**, not a new kind of
job: the swap runs exactly as it does for an uploaded photo, through the same
detector, guards and `FaceCompositor`. What a template adds is that the target
was chosen and verified in advance instead of supplied by whoever is using it.

Three consequences follow from being bundled, and they are the whole reason this
module is small:

- **No transfer.** Templates live on the pipeline's own filesystem, so
  `set_target` resolves them the way it was always meant to. The upload path
  photo mode needed does not apply.
- **Ambiguity is resolved at authoring time.** A scene with several people is
  refused by the multi-face guard, because "which face did you mean?" has no
  safe default. A template answers it once, offline, as `face_point` — so the
  guard has nothing left to protect against.
- **Failure is a build problem, not a runtime one.** `tools/validate_templates.py`
  runs the real guards over the library, so a template whose face is too small,
  too blurred or ambiguous never ships. A user should never meet a refusal
  caused by an asset we chose.

`face_point` is stored **normalised** and matched by proximity rather than as an
index into the detection list. Detection order is not a stable contract — it can
shift with a model pack — and an index that silently comes to mean a different
person is exactly the confidently-wrong output the guards exist to prevent. A
point keeps meaning the same thing.
"""

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from pipeline.logging import emit_status, emit_warning
from pipeline.types import Detection

# Manifest filename inside the templates directory.
MANIFEST_NAME = 'templates.json'


def resolve_templates_dir() -> str:
    """
    Where the template library lives.

    Checks the RunPod network volume first so the library survives pod restarts
    without being re-fetched, then the package directory — the same order the
    model weights use, and for the same reason.

    Returns:
        Absolute path to the templates directory (which may not exist yet)
    """
    if os.path.isdir('/workspace'):
        return os.path.join('/workspace', 'templates')
    package_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(package_dir, 'templates')


@dataclass(frozen=True)
class Template:
    """
    One bundled target scene.

    Attributes:
        id: Stable identifier, used by the API and never shown to a user
        name: Human label for the gallery
        image: Absolute path to the target photo
        face_point: Normalised (x, y) of the face to replace, or None to take
                    the largest — the same default every other path uses
        foreground: Absolute path to an optional RGBA layer composited *over*
                    the swap, for hair, glasses or a hand that should stay in
                    front of the face. Authored once and therefore more
                    reliable than asking XSeg to work it out per run
        thumbnail: Absolute path to a gallery thumbnail, or the image itself
        credit: Attribution or licence note for the asset
    """

    id: str
    name: str
    image: str
    face_point: Optional[Tuple[float, float]] = None
    foreground: Optional[str] = None
    thumbnail: Optional[str] = None
    credit: str = ''

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for the API. Paths stay server-side; clients get ids."""
        return {
            'id': self.id,
            'name': self.name,
            'face_point': list(self.face_point) if self.face_point else None,
            'has_foreground': self.foreground is not None,
            'credit': self.credit,
        }


class TemplateLibrary:
    """
    The bundled templates, read from a manifest beside the images.

    A missing or unreadable library is an empty one, not an error: templates are
    an addition to the product, and a pipeline without them still swaps
    uploaded photos exactly as before.
    """

    def __init__(self, directory: Optional[str] = None) -> None:
        """
        Args:
            directory: Where to look. Defaults to `resolve_templates_dir()`
        """
        self.directory = directory or resolve_templates_dir()
        self._templates: Dict[str, Template] = {}
        self._loaded = False

    def load(self) -> None:
        """Read the manifest. Safe to call repeatedly; reloads each time."""
        self._templates = {}
        self._loaded = True

        manifest_path = os.path.join(self.directory, MANIFEST_NAME)
        if not os.path.isfile(manifest_path):
            return

        try:
            with open(manifest_path, encoding='utf-8') as fh:
                data = json.load(fh)
        except (OSError, ValueError) as e:
            emit_warning(
                f'Template manifest unreadable ({type(e).__name__}: {e}) — '
                f'continuing without bundled templates',
                scope='TEMPLATES',
            )
            return

        for entry in data.get('templates', []):
            template = self._parse(entry)
            if template is not None:
                self._templates[template.id] = template

        if self._templates:
            emit_status(f'{len(self._templates)} template(s) available', scope='TEMPLATES')

    def _parse(self, entry: Dict[str, Any]) -> Optional[Template]:
        """
        Build one Template from a manifest entry, or None if it is unusable.

        An entry naming a file that is not there is skipped with a warning
        rather than failing the load: one broken asset must not cost the
        library, the same way one bad photo does not cost a photo job.
        """
        template_id = str(entry.get('id', '')).strip()
        image_name = str(entry.get('image', '')).strip()
        if not template_id or not image_name:
            emit_warning('Template entry missing id or image — skipped', scope='TEMPLATES')
            return None

        image = os.path.join(self.directory, image_name)
        if not os.path.isfile(image):
            emit_warning(f'Template {template_id}: image not found — skipped', scope='TEMPLATES')
            return None

        point = entry.get('face_point')
        face_point: Optional[Tuple[float, float]] = None
        if isinstance(point, (list, tuple)) and len(point) == 2:
            try:
                face_point = (float(point[0]), float(point[1]))
            except (TypeError, ValueError):
                emit_warning(
                    f'Template {template_id}: face_point is not a pair of numbers '
                    f'— falling back to the largest face',
                    scope='TEMPLATES',
                )

        return Template(
            id=template_id,
            name=str(entry.get('name', template_id)),
            image=image,
            face_point=face_point,
            foreground=self._optional_path(entry.get('foreground'), template_id),
            thumbnail=self._optional_path(entry.get('thumbnail'), template_id) or image,
            credit=str(entry.get('credit', '')),
        )

    def _optional_path(self, name: Any, template_id: str) -> Optional[str]:
        """Resolve an optional companion file, warning if it is named but absent."""
        if not name:
            return None
        path = os.path.join(self.directory, str(name))
        if not os.path.isfile(path):
            emit_warning(
                f'Template {template_id}: {name} not found — ignored',
                scope='TEMPLATES',
            )
            return None
        return path

    def all(self) -> List[Template]:
        """Every usable template, in manifest order."""
        if not self._loaded:
            self.load()
        return list(self._templates.values())

    def get(self, template_id: str) -> Optional[Template]:
        """One template by id, or None if the library does not have it."""
        if not self._loaded:
            self.load()
        return self._templates.get(template_id)


def select_by_point(
    detections: Sequence[Detection],
    point: Optional[Tuple[float, float]],
    frame_shape: Tuple[int, int],
) -> Optional[Detection]:
    """
    Pick the detection a normalised point falls in, or nearest to.

    Containment is tried first so an author clicking anywhere on the intended
    face is unambiguous. Nearest-centre is the fallback for a point that lands
    just outside a box — on the hairline, say — which should still resolve to
    the obvious face rather than to nothing.

    Args:
        detections: Faces found in the frame
        point: Normalised (x, y) in [0, 1], or None for no preference
        frame_shape: (height, width) of the frame the detections came from

    Returns:
        The chosen detection, or None when there is no point or no detections
    """
    if not detections or point is None:
        return None

    height, width = frame_shape[:2]
    target_x = point[0] * width
    target_y = point[1] * height

    for detection in detections:
        bbox = detection.bbox
        if (bbox.x <= target_x <= bbox.x + bbox.w
                and bbox.y <= target_y <= bbox.y + bbox.h):
            return detection

    def distance(detection: Detection) -> float:
        bbox = detection.bbox
        centre_x = bbox.x + bbox.w / 2.0
        centre_y = bbox.y + bbox.h / 2.0
        return (centre_x - target_x) ** 2 + (centre_y - target_y) ** 2

    return min(detections, key=distance)


def composite_foreground(frame: Any, foreground_path: str) -> Any:
    """
    Lay an authored RGBA layer over a finished swap.

    This is what puts hair, glasses or a hand back in front of the face. XSeg
    already does this from the frame itself, but a template's occlusion is
    fixed and known, so it can be drawn once by hand and be right every time
    rather than approximately right per run.

    A layer that cannot be read, or does not match the frame, is skipped: the
    swap without its foreground is imperfect, while a crash loses the job.

    Args:
        frame: BGR frame to draw over
        foreground_path: Path to a 4-channel PNG

    Returns:
        The composited frame, or the input unchanged if the layer is unusable
    """
    import cv2
    import numpy as np

    layer = cv2.imread(foreground_path, cv2.IMREAD_UNCHANGED)
    if layer is None:
        emit_warning(f'Foreground layer unreadable: {foreground_path}', scope='TEMPLATES')
        return frame

    if layer.shape[:2] != frame.shape[:2]:
        layer = cv2.resize(
            layer, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_AREA
        )

    if layer.ndim != 3 or layer.shape[2] != 4:
        emit_warning(
            f'Foreground layer has no alpha channel: {foreground_path}',
            scope='TEMPLATES',
        )
        return frame

    alpha = (layer[:, :, 3:4].astype(np.float32) / 255.0)
    colour = layer[:, :, :3].astype(np.float32)
    base = frame.astype(np.float32)
    return np.clip(colour * alpha + base * (1.0 - alpha), 0, 255).astype(np.uint8)
