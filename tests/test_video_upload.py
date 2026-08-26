"""
Chunked target-video upload: the limits, and the ways a transfer goes wrong.

Video targets exist because `set_target` resolves with `os.path.exists` against
the *pipeline's* filesystem, so a clip chosen on the desktop is simply absent
when the pipeline runs on a pod. What is checked here is not that a good file
arrives — that is one line — but that a bad one is refused at the earliest point
it can be, since every alternative fails later and says less.
"""
import base64
import os
import sys

from pipeline.config import FaceSwapConfig
from pipeline.api import handlers as H
from pipeline.api.schema import MAX_VIDEO_BYTES, MAX_VIDEO_SECONDS

FAIL = []
PASS = []


def check(label, condition, detail=''):
    """Record a check, printing it either way."""
    (PASS if condition else FAIL).append(label)
    print('  [{}] {}{}'.format(
        'PASS' if condition else 'FAIL', label, ' - ' + detail if detail else ''))


cfg = FaceSwapConfig()

print('\nRefused before any bytes move')

r = H.handle_upload_video_begin(cfg, 'big.mp4', MAX_VIDEO_BYTES + 1)
check('an oversized declaration is refused at begin', not r.success, r.error or '')
check('the refusal names the limit', str(MAX_VIDEO_BYTES // (1024 * 1024)) in (r.error or ''))

r = H.handle_upload_video_begin(cfg, 'notes.txt', 1000)
check('a non-video is refused at begin', not r.success, r.error or '')

r = H.handle_upload_video_begin(cfg, 'clip.mp4', 0)
check('a zero-size declaration is refused', not r.success, r.error or '')

print('\nA transfer that goes wrong stops at the chunk, not the end')

r = H.handle_upload_video_begin(cfg, 'clip.mp4', 1024)
check('a valid declaration opens an upload', r.success, str(r.data))
uid = (r.data or {}).get('upload_id', '')
check('begin hands back an upload id', bool(uid))

# One socket delivers in order, so a gap means a client bug or a stale retry.
# Accepting it would assemble a silently corrupt file, which surfaces as a
# render failing minutes later for no stated reason.
r = H.handle_upload_video_chunk(uid, 5, base64.b64encode(b'x').decode())
check('an out-of-order chunk is refused', not r.success, r.error or '')
check('and the upload is discarded rather than left open', uid not in H._VIDEO_UPLOADS)

r = H.handle_upload_video_begin(cfg, 'clip2.mp4', 1024)
uid2 = (r.data or {}).get('upload_id', '')
r = H.handle_upload_video_chunk(uid2, 0, '!!! not base64 !!!')
check('an undecodable chunk is refused', not r.success, r.error or '')
check('and that upload is discarded too', uid2 not in H._VIDEO_UPLOADS)

r = H.handle_upload_video_chunk('never-existed', 0, base64.b64encode(b'x').decode())
check('a chunk for an unknown upload is refused', not r.success, r.error or '')

print('\nAssembly checks what the client claimed')

# The bytes are the fact; the declaration was only a claim. A payload that is
# not a video assembles fine and must still be refused, because ffprobe cannot
# read a duration from it — and an unknown duration against a limit is not a
# pass.
r = H.handle_upload_video_begin(cfg, 'clip3.mp4', 32)
uid3 = (r.data or {}).get('upload_id', '')
H.handle_upload_video_chunk(uid3, 0, base64.b64encode(b'not a video at all').decode())
staged = H._VIDEO_UPLOADS.get(uid3, {}).get('path', '')
r = H.handle_upload_video_end(cfg, uid3)
check('a clip whose duration cannot be read is refused', not r.success, r.error or '')
check('the refused file is deleted', bool(staged) and not os.path.isfile(staged))
check('and the upload is forgotten', uid3 not in H._VIDEO_UPLOADS)

r = H.handle_upload_video_end(cfg, 'never-existed')
check('ending an unknown upload is refused', not r.success, r.error or '')

print('\nCancelling')

r = H.handle_upload_video_begin(cfg, 'clip4.mp4', 1024)
uid4 = (r.data or {}).get('upload_id', '')
partial = H._VIDEO_UPLOADS.get(uid4, {}).get('path', '')
H.handle_upload_video_chunk(uid4, 0, base64.b64encode(b'partial').decode())
r = H.handle_upload_video_cancel(uid4)
check('cancel succeeds', r.success)
check('cancel deletes the partial file', bool(partial) and not os.path.isfile(partial))

# A cancel racing a failure must not itself be an error.
check('cancel on an unknown id still succeeds',
      H.handle_upload_video_cancel('never-existed').success)

print('\nThumbnails')

r = H.handle_get_render_thumbnails(cfg)
check('the thumbnail handler answers', r.success, str(sorted((r.data or {}))))
# A handler that thumbnailed any path a client named would read arbitrary files
# off the pod. It answers only for what the config already points at.
check('it takes no path from the caller',
      'path' not in H.handle_get_render_thumbnails.__code__.co_varnames)
check('it reports both panes',
      'target' in (r.data or {}) and 'output' in (r.data or {}))

print('')
print('Getting the result back')

# A render writes on the pipeline's filesystem, so on a pod the operator cannot
# reach the file they just paid to produce. Reading it back is the other half of
# uploading, and refuses the same way when there is nothing there.
cfg.set('output_path', None)
r = H.handle_get_output_info(cfg)
check('output info is refused when nothing has been rendered',
      not r.success, r.error or '')
r = H.handle_get_output_chunk(cfg, 0, 1024)
check('an output chunk is refused when nothing has been rendered',
      not r.success, r.error or '')

import tempfile as _tf  # noqa: E402

_fd, _out = _tf.mkstemp(suffix='.mp4')
os.close(_fd)
with open(_out, 'wb') as _fh:
    _fh.write(b'0123456789' * 10)
cfg.set('output_path', _out)

r = H.handle_get_output_info(cfg)
check('output info reports the size',
      r.success and (r.data or {}).get('size') == 100, str(r.data))

r = H.handle_get_output_chunk(cfg, 0, 10)
check('a chunk reads from the offset given',
      r.success and base64.b64decode((r.data or {}).get('data', '')) == b'0123456789')
check('and says it is not the end', not (r.data or {}).get('eof', True))

r = H.handle_get_output_chunk(cfg, 90, 10)
check('the final chunk is marked eof', r.success and (r.data or {}).get('eof'))

r = H.handle_get_output_chunk(cfg, 500, 10)
check('an offset past the end is refused', not r.success, r.error or '')

# Neither takes a path. One that read any path a client named would serve
# arbitrary files off the pod.
for _fn in (H.handle_get_output_info, H.handle_get_output_chunk):
    check('{} takes no path from the caller'.format(_fn.__name__),
          'path' not in _fn.__code__.co_varnames[:_fn.__code__.co_argcount])

os.remove(_out)
cfg.set('output_path', None)


print('\nThe two limits are different limits')

# Bytes bound the transfer, seconds bound the render, and they do not track each
# other: a well compressed ten minutes can be smaller than a badly compressed
# one while costing twenty times as much to process.
check('both limits exist and are independent',
      MAX_VIDEO_BYTES > 0 and MAX_VIDEO_SECONDS > 0)

print('\n{} passed, {} failed'.format(len(PASS), len(FAIL)))


def test_everything_passed() -> None:
    """Surface the checks above to pytest as one assertion."""
    assert not FAIL, '{} of {} checks failed: {}'.format(
        len(FAIL), len(PASS) + len(FAIL), ', '.join(FAIL))


if __name__ == '__main__':
    sys.exit(1 if FAIL else 0)
