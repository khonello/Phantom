"""
Async wrapper for frame processors.

Wraps any FrameProcessor and runs it in a background thread with a queue.
Useful for expensive operations that should not block the main pipeline loop.

Extracted from stream.py:_enhancement_worker().
"""

import queue
import threading
from typing import Optional, Tuple

from pipeline.processing.frame_processor import FrameProcessor
from pipeline.types import Frame


class AsyncProcessor:
    """
    Wraps a processor to run asynchronously in a background thread.

    The processor is fed frames via submit() and outputs can be retrieved
    with get_latest(). The queue is always size 1 to keep only the latest
    frame (older frames are dropped if not consumed).

    Example:
        async_proc = AsyncProcessor(enhancer_proc, stop_event)
        async_proc.start()
        async_proc.submit(seq=1, frame=frame1)
        seq, enhanced = async_proc.get_latest()
        async_proc.stop()
        async_proc.join()
    """

    def __init__(self, processor: FrameProcessor, stop_event: threading.Event) -> None:
        """
        Initialize async processor.

        Args:
            processor: FrameProcessor to run asynchronously
            stop_event: threading.Event to signal shutdown
        """
        self.processor = processor
        self.stop_event = stop_event

        # Queues are size 3 to allow burst smoothing while still dropping
        # under sustained overload. Size 1 caused too many silent drops.
        self._input_queue: queue.Queue = queue.Queue(maxsize=3)
        self._output_queue: queue.Queue = queue.Queue(maxsize=3)

        self._thread: Optional[threading.Thread] = None
        self._running = False

        # Drop counter for backpressure visibility
        self.drop_count: int = 0

    def submit(self, seq: int, frame: Frame) -> None:
        """
        Submit a frame for processing.

        If queue is full, drops oldest frame (keeps only latest).

        Args:
            seq: Sequence number (for tracking frame order)
            frame: Frame to process
        """
        # Drop old frame if queue full and record metric
        if self._input_queue.full():
            try:
                self._input_queue.get_nowait()
                self.drop_count += 1
            except queue.Empty:
                pass

        # Add new frame
        try:
            self._input_queue.put_nowait((seq, frame))
        except queue.Full:
            pass

    def get_latest(self) -> Optional[Tuple[int, Frame]]:
        """
        Get latest processed frame (non-blocking).

        Returns:
            Tuple of (seq, frame) or None if no output available
        """
        try:
            return self._output_queue.get_nowait()
        except queue.Empty:
            return None

    def start(self) -> None:
        """Start the worker thread."""
        if self._thread is not None:
            return

        self._running = True
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Signal worker thread to stop."""
        self._running = False

    def join(self, timeout: float = 2.0) -> None:
        """
        Wait for worker thread to finish.

        Args:
            timeout: Max wait time in seconds
        """
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None

    def _worker(self) -> None:
        """Worker thread main loop (runs processor asynchronously)."""
        while not self.stop_event.is_set() and self._running:
            try:
                # Get input with short timeout to allow checking stop_event
                seq, frame = self._input_queue.get(timeout=0.01)
            except queue.Empty:
                continue

            # Process frame
            try:
                processed = self.processor.process(frame)
            except Exception as e:
                import sys
                print(f'[AsyncProcessor] frame processing error (seq={seq}): {type(e).__name__}: {e}', file=sys.stderr)
                processed = frame

            # Put output (drop old result if queue full)
            if self._output_queue.full():
                try:
                    self._output_queue.get_nowait()
                except queue.Empty:
                    pass

            try:
                self._output_queue.put_nowait((seq, processed))
            except queue.Full:
                pass
