"""
Wrap video ffmpeg video decoding with a TensorFlow python op.

See usage_example.py for details on how to use the python op.
"""

from __future__ import absolute_import

import random
import numpy as np
import tensorflow as tf

from ffmpeg_reader import FFMPEG_VideoReader


def _load_video_ffmpeg(filename, n_frames, target_fps, random_chunk):
    """
    Load a video as a numpy array using FFmpeg in [0, 255] RGB format.
    :param filename: path to the video file
    :param n_frames: number of frames to decode
    :param target_fps: framerate at which the video will be decoded
    :param random_chunk: grab frames starting from a random position
    :return: (video, length) tuple
        video: (n_frames, h, w, 3) numpy array containing video frames, as RGB in range [0, 255]
        height: frame height
        width: frame width
        length: number of non-zero frames loaded from the video (the rest of the sequence is zero-padded)
    """
    # Make sure that the types are correct
    if isinstance(filename, bytes):
        filename = filename.decode('utf-8')
    if isinstance(n_frames, np.int32):
        n_frames = int(n_frames)

    # Get video params
    video_reader = FFMPEG_VideoReader(filename, target_fps=target_fps)
    w, h = video_reader.size
    fps = video_reader.fps
    if target_fps <= 0:
        target_fps = fps
    video_length = int(video_reader.nframes * target_fps / fps)  # corrected number of frames
    tensor_shape = [n_frames, h, w, 3]

    # Determine starting and ending positions
    if n_frames <= 0:
        n_frames = video_length
        tensor_shape[0] = video_length
    elif video_length < n_frames:
        n_frames = video_length
    elif random_chunk:  # start from a random position
        start_pos = random.randint(0, video_length - n_frames - 1)
        video_reader.get_frame(1. * start_pos / target_fps, fps=target_fps)

    # Load video chunk as numpy array
    video = np.zeros(tensor_shape, dtype=np.float32)
    for idx in range(n_frames):
        video[idx, :, :, :] = video_reader.read_frame()[:, :, :3].astype(np.float32)

    video_reader.close()

    return video, h, w, n_frames


def decode_video(filename, n_frames=0, fps=-1, random_chunk=False):
    """
    Decode frames from a video. Returns frames in [0, 255] RGB format.
    :param filename: string tensor, e.g. dequeue() op from a filenames queue
    :return:
        video: 4-D tensor containing frames of a video: [time, height, width, channel]
        height: frame height
        width: frame width
        length: number of non-zero frames loaded from the video (the rest of the sequence is zero-padded)
    """
    params = [filename, n_frames, fps, random_chunk]
    dtypes = [tf.float32, tf.int64, tf.int64, tf.int64]
    return tf.py_func(_load_video_ffmpeg, params, dtypes, name='decode_video')
