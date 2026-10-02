"""Run `lerobot-train` with every PyAV video decoder pinned to a single thread.

LeRobot's PyAV backend opens a fresh container for every frame it decodes. With PyAV's
default thread_count=0, FFmpeg/libdav1d spins up one decoder thread per visible core on
each open, which for AV1 datasets (e.g. g1-carry-box) makes decoding 3-7x slower than a
single thread and starves the GPU. Each DataLoader worker is already its own process, so
single-threaded decoders lose nothing.

Usage: python tools/lerobot_train_pyav_single_thread.py <lerobot-train args...>
"""

import av

_av_open = av.open


def _single_thread_open(*args, **kwargs):
    container = _av_open(*args, **kwargs)
    if isinstance(container, av.container.InputContainer):
        for stream in container.streams.video:
            stream.codec_context.thread_count = 1
    return container


# Patch at import time (not under __main__) so spawned DataLoader workers, which re-import
# this module as __mp_main__, get the patch too.
av.open = _single_thread_open

if __name__ == "__main__":
    from lerobot.scripts.lerobot_train import main

    main()
