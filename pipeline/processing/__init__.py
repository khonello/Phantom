"""
Processing pipeline for the Phantom face-swapping application.

Composes frame processors into a processing pipeline that handles:
- Face detection and tracking
- Face swapping
- Enhancement
- Blending
- Output

Replaces the monolithic stream.py:_pipeline_loop() with composable,
testable, event-driven processors.
"""
