"""
The sidebar can be scrolled to everything it contains.

The action button — START, or PROCESS — sits at the *bottom* of the mode pane,
so it is the first thing to go when the window is shorter than the sidebar
wants to be. Wrapping the column in a Flickable is not by itself enough, and
the way it failed is worth keeping a test around for:

each mode pane is an `Item` holding a `ColumnLayout` with `anchors.fill:
parent`, and **an anchored child contributes nothing to its parent's
implicitHeight**. The pane therefore measured 0, the sidebar column's implicit
height came out as just the few controls above the panes, and
`Math.max(implicitHeight, viewport)` picked the viewport at every window size.
`contentHeight` equalled `height`, the Flickable had nothing to scroll, and the
button stayed exactly as unreachable as it had been before there was a
Flickable at all. It looked fixed and was not.

So this measures rather than inspects: load the window at a range of heights
and assert two things per size — the content grows past the viewport when it
needs to (and not when it does not), and no visible item sits below
`contentHeight`, where scrolling cannot reach it.

Skipped without PySide6, which the CI interpreter does not have. Run it against
the Qt environment in `desktop/.qtcreator/`.
"""

import os
import sys
# conftest.py handles this under pytest; this covers `python tests/<file>.py`.
import os as _os
_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import pytest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

pytest.importorskip('PySide6', reason='desktop layout needs Qt; see desktop/.qtcreator/')

from PySide6.QtCore import (  # noqa: E402
    Property, QObject, QTimer, QUrl, Signal,
)
from PySide6.QtGui import QGuiApplication  # noqa: E402
from PySide6.QtQml import (  # noqa: E402
    QQmlApplicationEngine, QQmlPropertyMap, qmlRegisterType,
)
from PySide6.QtQuick import QQuickItem  # noqa: E402

QML = os.path.join(_REPO_ROOT, 'desktop', 'main.qml')

# Enough of the bridge for every binding the sidebar reads. A missing property
# is not harmless here: the binding errors, `visible` falls back to its default
# of true, and *both* mode panes get laid out at once — which silently inflates
# the measurement by a whole pane.
_STATE = {
    'currentMode': 'realtime', 'mediaTab': 'video',
    'connectionLabel': 'ws://localhost:9000/ws', 'statusMessage': 'idle',
    'sourceLabel': 'face.jpg', 'sourceThumbnail': '', 'targetLabel': 'clip.mp4',
    'targetThumbnail': '', 'outputPath': '/tmp/out.mp4', 'outputThumbnail': '',
    'detectionStatus': '', 'guardReason': '', 'latencyText': '',
    'loadingMessage': '', 'restoration': 'auto', 'activeFilter': 'none',
    'activeEffect': 'none', 'sessionReason': '', 'authError': '',
    'pickerPhoto': '',
    'connected': True, 'pipelineRunning': False, 'batchRunning': False,
    'batchComplete': False, 'sourceSet': True, 'targetSet': True,
    'embeddingPending': False, 'virtualCamActive': True,
    'filtersEnabled': False, 'filterPanel': False, 'faceNoticeOpen': False,
    'pickerOpen': False, 'photoNeedsFace': False, 'sessionExpired': False,
    'authRequired': False, 'authChecking': False, 'statusError': False,
    'liveVersion': 0, 'webcamVersion': 0, 'authMinutes': 60,
    'maxPhotoTargets': 4, 'pickerTotal': 0, 'selectedTemplate': -1,
    'filterList': [], 'effectList': [], 'templates': [], 'photoTargets': [],
    'photoResults': [], 'pickerBoxes': [], 'pickerPosition': [],
}


class _FrameDisplay(QQuickItem):
    """
    Stands in for the type `desktop/bridge.py` registers from Python.

    The properties are not decoration: `main.qml` assigns both, and assigning to
    a property a registered type does not have is a *load* error, so a bare
    QQuickItem here fails the whole window rather than just the viewport.
    """

    sourceChanged = Signal()
    frameVersionChanged = Signal()

    def __init__(self, parent: object = None) -> None:
        super().__init__(parent)
        self._source = ''
        self._frame_version = 0

    def _get_source(self) -> str:
        return self._source

    def _set_source(self, value: str) -> None:
        self._source = value

    def _get_frame_version(self) -> int:
        return self._frame_version

    def _set_frame_version(self, value: int) -> None:
        self._frame_version = value

    source = Property(str, _get_source, _set_source, notify=sourceChanged)
    frameVersion = Property(
        int, _get_frame_version, _set_frame_version, notify=frameVersionChanged,
    )


def _measure(height: int, mode: str) -> dict:
    """Load the window at one size and report what the sidebar did."""
    app = QGuiApplication.instance() or QGuiApplication([])
    qmlRegisterType(_FrameDisplay, 'Phantom', 1, 0, 'FrameDisplay')

    bridge = QQmlPropertyMap()
    for name, value in _STATE.items():
        bridge.insert(name, value)
    bridge.insert('currentMode', mode)
    bridge.insert('mediaTab', 'video' if mode in ('realtime', 'video') else 'image')

    engine = QQmlApplicationEngine()
    # `setContextProperty` does not take ownership: a local would be collected,
    # `bridge` would go null, and every binding reading it would error out to
    # its default — which is how both panes came to be measured at once.
    engine.rootContext().setContextProperty('bridge', bridge)
    engine.load(QUrl.fromLocalFile(QML))

    roots = engine.rootObjects()
    assert roots, 'main.qml produced no root object'
    window = roots[0]
    window.setProperty('width', 1600)
    window.setProperty('height', height)

    result: dict = {}

    def collect() -> None:
        flickable = next(
            (c for c in window.findChildren(QObject)
             if c.metaObject().className().startswith('QQuickFlickable')),
            None,
        )
        assert flickable is not None, 'the sidebar has no Flickable'
        content = flickable.property('contentItem')

        deepest = 0.0

        def walk(item: QQuickItem) -> None:
            nonlocal deepest
            for kid in item.childItems():
                if not kid.property('visible'):
                    continue
                kid_height = kid.property('height') or 0
                if kid_height > 0:
                    deepest = max(deepest, kid.mapToItem(content, 0, kid_height).y())
                walk(kid)

        walk(content)
        result.update(
            viewport=float(flickable.property('height')),
            content=float(flickable.property('contentHeight')),
            deepest=float(deepest),
        )
        app.quit()

    QTimer.singleShot(120, collect)
    QTimer.singleShot(8000, app.quit)   # never hang the suite
    app.exec()

    engine.deleteLater()
    assert result, 'the window never laid out'
    return result


@pytest.mark.parametrize('height', [900, 700, 620, 500, 400])
@pytest.mark.parametrize('mode', ['realtime', 'video'])
def test_nothing_sits_below_the_scrollable_region(height: int, mode: str) -> None:
    """
    Everything in the sidebar is reachable by scrolling to it.

    This is the property the operator actually cared about: the button exists
    and can be got to, at whatever height the window happens to be.
    """
    measured = _measure(height, mode)
    assert measured['deepest'] <= measured['content'] + 1, (
        '{} at {}px: content is {:.0f} but something reaches {:.0f} — '
        '{:.0f}px below where scrolling can go'.format(
            mode, height, measured['content'], measured['deepest'],
            measured['deepest'] - measured['content'])
    )


@pytest.mark.parametrize('mode', ['realtime', 'video'])
def test_the_content_grows_past_a_short_viewport(mode: str) -> None:
    """
    The Flickable has something to scroll when the window is too short.

    The regression this exists for passed the check above trivially — with
    `contentHeight` pinned to the viewport, nothing was ever below it, because
    the overflow was being clipped rather than made scrollable.
    """
    tall = _measure(900, mode)
    short = _measure(400, mode)

    assert tall['content'] <= tall['viewport'] + 1, (
        'a tall window should not scroll: content {:.0f} > viewport {:.0f}'
        .format(tall['content'], tall['viewport'])
    )
    assert short['content'] > short['viewport'] + 1, (
        '{} at 400px: content {:.0f} is not past viewport {:.0f} — the panes '
        'are reporting no implicit height again, so Math.max always picks the '
        'viewport and the Flickable has nothing to scroll'.format(
            mode, short['content'], short['viewport'])
    )
