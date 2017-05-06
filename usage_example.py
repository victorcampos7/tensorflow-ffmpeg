"""
Usage example for the decode_video python op.
"""


from __future__ import absolute_import
from __future__ import print_function

import time
import argparse
import numpy as np
import tensorflow as tf

from py_ops import decode_video


def _parse_arguments():
    parser = argparse.ArgumentParser('Test decode_video python op.')
    parser.add_argument('--input_file', help='Path to the video file.')
    parser.add_argument('--output_file', default=None, 
    	help='(Optional) Path to the .npy file where the decoded frames will be stored.')
    parser.add_argument('--play_video', default=False, action='store_true',
                        help='Play the extracted frames.')
    parser.add_argument('--num_frames', default=30, type=int, 
    	help='Number of frames per video (sequence length). Set to 0 for full video.')
    parser.add_argument('--fps', default=-1, type=int, 
    	help='Framerate to which the input videos are converted. Use -1 for the original framerate.')
    parser.add_argument('--random_chunks', default=False, action='store_true',
    	help='Grab video frames starting from a random position.')
    return parser.parse_args()


def _show_video(video, fps=10):
    # Import matplotlib/pylab only if needed
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pylab as pl
    pl.style.use('ggplot')
    pl.axis('off')

    if fps < 0:
        fps = 25
    video /= 255.  # Pylab works in [0, 1] range
    img = None
    pause_length = 1. / fps
    try:
        for f in range(video.shape[0]):
            im = video[f, :, :, :]
            if img is None:
                img = pl.imshow(im)
            else:
                img.set_data(im)
            pl.pause(pause_length)
            pl.draw()
    except:
        pass


if __name__ == '__main__':
    args = _parse_arguments()
    sess = tf.Session()
    f = tf.placeholder(tf.string)
    video, h, w, seq_length = decode_video(f, args.num_frames, args.fps, args.random_chunks)
    start_time = time.time()
    frames, seq_length_val = sess.run([video, seq_length], feed_dict={f: args.input_file})
    total_time = time.time() - start_time
    print('\nSuccessfully loaded video!\n'
          '\tDimensions: %s\n'
          '\tTime: %.3fs\n'
          '\tLoaded frames: %d\n' %
          (str(frames.shape), total_time, seq_length_val))
    if args.output_file:
    	np.save(args.output_file, frames)
        print("Stored frames to %s" % args.output_file)
    if args.play_video:
        _show_video(frames, args.fps)
