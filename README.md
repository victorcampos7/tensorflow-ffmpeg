# Loading videos in TensorFlow with FFmpeg

TensorFlow does not offer any native operation to decode videos inside the graph. This issue is usually overcome by first extracting and storing video frames as individual images and then loading them using TensorFlow's native image ops. Such approach, however, is not scalable and becomes prohibitive for large video collections. This repository contains a python op that wraps FFmpeg and allows to decode and load videos directly inside the graph.


## Requirements

This code has been tested with [TensorFlow](https://www.tensorflow.org) 1.0. [FFmpeg](https://ffmpeg.org) is required in order to perform video decoding.


## Usage

Video can be decoded through the *decode_video()* python op, similarly to how an image would be decoded through *[tf.image.decode_jpeg()](https://www.tensorflow.org/api_docs/python/tf/image/decode_jpeg)*. While images are decoded into 3D tensors with shape (H, W, D), videos are decoded into 4D tensors with shape (T, H, W, D).

Please see `usage_example.py` for an example on how to use the custom op. If FFmpeg is not found, make sure that the proper path to the binary is set either by modifying the *FFMPEG_BIN* variable or calling *set_ffmpeg_bin()* in `ffmpeg_reader.py`.


## References

The code for decoding videos using FFmpeg within Python is a modification of the one in the [moviepy project](https://github.com/Zulko/moviepy).
