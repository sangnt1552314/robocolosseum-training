"""Left-eye crop for the Unitree G1 head camera, shared by dataset conversion and inference.

The G1 head camera delivers a 480x1280 side-by-side stereo frame: left eye in columns 0-639,
right eye in columns 640-1279. Policies trained on g1-carry-box-left-eye see only the left eye
(480x640, the same size as the wrist cameras). At inference, pass the robot's raw head frame
through `crop_head_left_eye` before building `observation.images.head`, exactly as the training
data was made.
"""

STEREO_WIDTH = 1280
EYE_WIDTH = STEREO_WIDTH // 2


def crop_head_left_eye(frame, channels_first: bool = False):
    """Return the left-eye half of a side-by-side stereo head frame.

    Works on numpy arrays and torch tensors, with or without leading batch/time dims.
    `frame` is (..., H, W, C) by default, or (..., C, H, W) with channels_first=True.
    """
    width_axis = -1 if channels_first else -2
    width = frame.shape[width_axis]
    if width != STEREO_WIDTH:
        raise ValueError(
            f"Expected a {STEREO_WIDTH}-wide side-by-side stereo head frame, got width {width} "
            f"(shape {tuple(frame.shape)}). Was it already cropped?"
        )
    if channels_first:
        return frame[..., :EYE_WIDTH]
    return frame[..., :EYE_WIDTH, :]
