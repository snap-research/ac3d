import os
import re
import sys
import json
import yaml
import logging
import datetime
from typing import Any, Dict
from collections import OrderedDict
import time
from typing import List, BinaryIO, Union
import traceback

from loguru import logger
from beartype import beartype
import safetensors
from PIL import Image
import torch
import av
from PIL import Image

#----------------------------------------------------------------------------
# Some constants.

VIDEO_EXTENSIONS = {'.avi', '.mp4', '.webm', '.mov', '.mkv', '.m4v', '.mpg', '.mpeg', '.wmv', '.flv', '.3gp', '.3g2'}

#----------------------------------------------------------------------------
# OS utils.

def file_ext(path: os.PathLike) -> str:
    return os.path.splitext(path)[1].lower()

def save_yaml(data, path: os.PathLike) -> None:
    with open(path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, width=999999999, sort_keys=False)

def load_yaml(path: os.PathLike) -> Any:
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def load_safetensors_state(p: str) -> Dict[str, torch.Tensor]:
    assert os.path.isfile(p), f"File not found: {p}"
    tensors = {}
    with safetensors.safe_open(p, framework="pt", device=0) as f:
        for k in f.keys():
            tensors[k] = f.get_tensor(k)
    return tensors

@beartype
def save_json(data: dict, p: str, **kwargs) -> None:
    with open(p, "w") as f:
        json.dump(data, f, **kwargs)

@beartype
def load_json(p: str) -> dict:
    with open(p, "r") as f:
        return json.load(f)

def find_images_in_dir(dir_path: os.PathLike, **kwargs):
    Image.init()
    return find_files_in_dir(dir_path, Image.EXTENSION, **kwargs)

def find_videos_in_dir(dir_path: os.PathLike, **kwargs):
    return find_files_in_dir(dir_path, VIDEO_EXTENSIONS, **kwargs)

def find_files_in_dir(dir_path: os.PathLike, extensions, ignore_regex=None, full_path: bool=True):
    files_all = [os.path.relpath(os.path.join(root, fname), start=dir_path) for root, _dirs, files in os.walk(dir_path) for fname in files]
    files_filtered = sorted([f for f in files_all if file_ext(f) in extensions])
    if full_path:
        files_filtered = [os.path.join(dir_path, f) for f in files_filtered]
    if not ignore_regex is None:
        files_filtered = [f for f in files_filtered if not re.fullmatch(ignore_regex, f)]
    return files_filtered


class VideoDecoder:
    """
    Utility class for decoding videos
    """
    def __init__(self, file: Union[str, BinaryIO], default_thread_type: str=None, enable_frame_parallel_decoding: bool = False, gop_size_hint_forward: int=10, gop_size_hint_backward: int=5):
        """
        :param file: The video file to decode
        :param default_thread_type: The default thread type to use for decoding the video.
                                    The use of threading may reduce performance when decoding single, randomly-accessed frames
        :param enable_frame_parallel_decoding: Whether to use multiple threads for decoding multiple frames
                                               May cause excessive thread creation if video-level parallelism is used already
        :param gop_size_hint_forward: The number of frames that is acceptable to decode forward in time to reach the frame of interest rather than performing a seek operation
                                      Seek is an expensive operation, so it may be beneficial to set the parameter to a value larger than the gop size if the gop size is small, eg. 5 frames
        :param gop_size_hint_backward: The number of frames that we will go backward in time if the seek overshoots the target

        """
        self.file = file
        if isinstance(file, str):
            self.file = open(file, "rb")

        # Some video metadata may break pyav for unknown reasons (https://github.com/PyAV-Org/PyAV/issues/629)
        # We ignore errors in metadata loading. Even with this option it seems the other metadata remain readable
        self.container = av.open(file, mode="r", metadata_errors="ignore")

        self.video_stream = self.container.streams.video[0]
        self.framerate = float(self.video_stream.guessed_rate)  # use guessed_rate which is more robust than codec_context.framerate which only looks at a few frames
        self.frame_duration = 1 / self.framerate

        # The default threading value used by the video stream
        self.default_video_stream_thread_type = default_thread_type
        if self.default_video_stream_thread_type is None:
            self.default_video_stream_thread_type = self.video_stream.thread_type
        self.video_stream.thread_type = self.default_video_stream_thread_type

        # Enabling this slows down decoding for a single frame, but can accelerate decoding of multiple frames
        # Linked to the use of
        # video_stream.thread_type = "AUTO"
        self.enable_frame_parallel_decoding = enable_frame_parallel_decoding

        # Seeks only if at least a number of frames greater than the GOP size is present between the current and next frame
        # Better to decode more subsequent frames than risk overshooting with a seek
        self.min_seek_time_interval = gop_size_hint_forward / self.framerate

        # If the seek overshoots the target, we go backwards in time of a common GOP size
        self.backward_search_time_step = gop_size_hint_backward / self.framerate

    def close(self):
        self.container.close()
        self.file.close()

    def decode_frame_at_index(self, index: int, frame_seek_timeout_sec: float=10.0) -> Image.Image:
        """
        Extracts a frame at the given index
        :param index: The index of the frame to read. The index is expressed assuming constant framerate and no missing frames
        :param frame_seek_timeout_sec: The maximum time to wait for a frame to be decoded. If the timeout is reached, the decoding is aborted and an exception is raised.
                                       Acts as a safeguard against unforeseen corruptions of the underlying media file
        :return: PIL image with the decoded frame
        """
        timestamp = index / self.framerate
        return self.decode_frame_at_time(timestamp, frame_seek_timeout_sec)

    def decode_frame_at_time(self, timestamp: int, frame_seek_timeout_sec: float=10.0) -> Image.Image:
        """
        Extracts a frame at the given timestamp
        :param timestamp: Timestamp of the frame to read
        :param frame_seek_timeout_sec: The maximum time to wait for a frame to be decoded. If the timeout is reached, the decoding is aborted and an exception is raised.
                                       Acts as a safeguard against unforeseen corruptions of the underlying media file
        :return: PIL image with the decoded frame
        """
        results = self.decode_frames_at_times([timestamp], frame_seek_timeout_sec)
        return results[0]

    def decode_frames_at_indexes(self, indexes: List[int], frame_seek_timeout_sec: float=10.0) -> List[Image.Image]:
        """
        Extracts the frames corresponding to the given indexes
        :param indexes: The indexes of the frames to read. The indexes are expressed assuming constant framerate and no missing frames
        :param frame_seek_timeout_sec: The maximum time to wait for a frame to be decoded. If the timeout is reached, the decoding is aborted and an exception is raised.
                                       Acts as a safeguard against unforeseen corruptions of the underlying media file
        :return: List of PIL images corresponding to the decoded frames
        """
        timestamps = [index / self.framerate for index in indexes]
        return self.decode_frames_at_times(timestamps, frame_seek_timeout_sec)

    def decode_frames_at_times(self, timestamps: List[float], frame_seek_timeout_sec: float=10.0) -> List[Image.Image]:
        """
        Extracts the frames corresponding to the given timestamps
        :param timestamps: The timestamps of the frames to read
        :param frame_seek_timeout_sec: The maximum time to wait for a frame to be decoded. If the timeout is reached, the decoding is aborted and an exception is raised.
                                       Acts as a safeguard against unforeseen corruptions of the underlying media file
        :return: List of PIL images corresponding to the decoded frames
        """
        # If more than one frame needs to be decoded and frame parallelism is allowed, we enable it
        # Frame parallelism introduces delay if used to read only a limited number of frames
        if not self.video_stream.codec_context.is_open: # We can change the thread type only of the stream is not open already
            if len(timestamps) > 1 and self.enable_frame_parallel_decoding:
                self.video_stream.thread_type = "AUTO"
            else:
                self.video_stream.thread_type = self.default_video_stream_thread_type

        # Sorts the timestamps
        timestamp_with_index = [(timestamp, index) for index, timestamp in enumerate(timestamps)]
        timestamp_with_index.sort(key=lambda x: x[0])

        decoded_images = []

        # Iterator that iterates the files being decoded. Necessary to keep the iterator open between different iterations as calling self.container.decode(video=0) multiple times can generate EOF errors
        decoding_iterator = None
        last_decoded_timestamp = float("-inf")
        for current_target_timestamp, original_order_index in timestamp_with_index:
            timeout_timer = time.time()

            # Special case where timestamps corresponding to the same frame are requested. Recycles the last decoded frame
            if abs(last_decoded_timestamp - current_target_timestamp) <= self.frame_duration / 2 + 0.001:
                last_frame_copy = decoded_images[-1][0].copy() # (frame, original_order_index)
                decoded_images.append((last_frame_copy, original_order_index))
                continue

            current_search_start_time = current_target_timestamp
            # Loop to adjust imprecise seek location
            found = False
            while not found:

                # Seeks only if at least a number of frames greater than a large GOP size is present between the current and next frame
                # Or if the frame we need to read is behind us
                # last_decoded_timestamp >= current_target_timestamp needed to reinitialize the iterator since the same frame cannot be read more than once without seeking again and reopining the iterator
                if current_target_timestamp - last_decoded_timestamp >= self.min_seek_time_interval or last_decoded_timestamp >= current_target_timestamp:
                    seek_offset = round(current_search_start_time / self.video_stream.time_base)
                    self.container.seek(seek_offset, backward=True, any_frame=False, stream=self.video_stream)
                    decoding_iterator = None # need to open a new iterator for decoding at seek

                try:
                    # Reads frames
                    loop_break_exit = False
                    if decoding_iterator is None:
                        decoding_iterator = iter(self.container.decode(video=0))
                        minimum_time_encountered_frame_from_last_seek = float("inf") # Detects whether a frame is missing if the target is between min and current and yet we have not found the frame
                        previous_decoded_timestamp = None
                    while True:
                        current_timeout_timer = time.time()
                        if current_timeout_timer - timeout_timer > frame_seek_timeout_sec:
                            raise Exception("Timeout of {}s reached while decoding frame at time {} in the current video {}.".format(frame_seek_timeout_sec, current_target_timestamp, self.file))

                        frame = next(decoding_iterator)
                    #for frame in self.container.decode(video=0):
                        last_decoded_timestamp = frame.time  # The timestamp of the currently read frame
                        # Frames are unordered, raise an exception
                        if previous_decoded_timestamp is not None and last_decoded_timestamp < previous_decoded_timestamp:
                            raise Exception("Frames in the video are unordered, the file may be corrupted. When opening the file the decoder may have logged 'CTTS invalid' indicating invalid mapping between decoding and presentation timestamps, leading to lack of ordering. This is not supported")
                        previous_decoded_timestamp = last_decoded_timestamp
                        minimum_time_encountered_frame_from_last_seek = min(minimum_time_encountered_frame_from_last_seek, last_decoded_timestamp)

                        # We found a frame that is closer than half the distance between each frame, so it is the closest to the target we can get
                        # Timestamps may be rounded, so we add a small epsilon to account for expanded intervals between frames due to rounding
                        if abs(last_decoded_timestamp - current_target_timestamp) <= self.frame_duration / 2 + 0.001:
                            found = True
                            loop_break_exit = True
                            break
                        # If we scanned from a frame that was lower than the target to a frame that is greater than the target without finding the target, then it means that the target is missing
                        elif last_decoded_timestamp > current_target_timestamp and minimum_time_encountered_frame_from_last_seek < current_target_timestamp:
                            raise Exception("Could not find the frame at time {} between the seek timestamp {} and the current timestamp {} in the current video {}. The video frame may be missing of video frames may be unordered. When opening the file the decoder may have logged 'CTTS invalid' indicating invalid mapping between decoding and presentation timestamps, leading to lack of ordering.".format(current_target_timestamp, minimum_time_encountered_frame_from_last_seek, last_decoded_timestamp, self.file))
                        # The seek overshoot the target
                        elif last_decoded_timestamp > current_target_timestamp:
                            loop_break_exit = True
                            break
                except StopIteration:
                    # If reading the last frame, the right interval covered by the last frame is not frame_duration / 2, but frame duration
                    if current_target_timestamp > last_decoded_timestamp and abs(last_decoded_timestamp - current_target_timestamp) <= self.frame_duration + 0.001:
                        found = True
                        loop_break_exit = True
                    else:
                        continue
                except Exception as e: # pylint: disable=broad-except
                    print_message = f"An exception occurred in VideoDecoder while decoding frames in {self.file}, "
                    exception_message = str(e)
                    exception_trace_string = traceback.format_exc()
                    print_message += "Exception message: {}\n".format(exception_message)
                    print_message += "Exception trace:\n{}".format(exception_trace_string)
                    print(print_message, file=sys.stderr, flush=True)

                # Frame not found
                if not found and (current_search_start_time == 0 or not loop_break_exit):
                    raise Exception("Could not find frame at time {} in the current video {}".format(current_target_timestamp, self.file))

                # If we overshoot the target with the seek, we go backwards in time for the next seek
                current_search_start_time -= self.backward_search_time_step
                if current_search_start_time < 0:
                    current_search_start_time = 0

            decoded_images.append((frame.to_image(), original_order_index))

        # Sorts back the results in the initial order
        decoded_images.sort(key=lambda x: x[1])
        decoded_images = [image for image, _ in decoded_images]

        return decoded_images

#----------------------------------------------------------------------------
# Distributed utils.

def init(disable_c10d_logging: bool=True, timeout_seconds: int=None):
    if disable_c10d_logging:
        logging.getLogger('torch.distributed.distributed_c10d').setLevel(logging.WARNING)

    if 'MASTER_ADDR' not in os.environ:
        os.environ['MASTER_ADDR'] = 'localhost'
    if 'MASTER_PORT' not in os.environ:
        os.environ['MASTER_PORT'] = '29500'
    if 'RANK' not in os.environ:
        os.environ['RANK'] = '0'
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = '0'
    if 'WORLD_SIZE' not in os.environ:
        os.environ['WORLD_SIZE'] = '1'

    backend = 'gloo' if os.name == 'nt' else 'nccl'
    timeout_kwargs = {} if timeout_seconds is None else dict(timeout=datetime.timedelta(seconds=timeout_seconds))
    # torch.distributed.init_process_group(backend=backend, init_method='env://', **timeout_kwargs)
    # torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', '0')))

def get_rank():
    return torch.distributed.get_rank() if torch.distributed.is_initialized() else int(os.environ.get('RANK', '0'))

def get_local_rank():
    return int(os.environ['LOCAL_RANK']) if torch.distributed.is_initialized() and 'LOCAL_RANK' in os.environ else 0

def is_main_process():
    return get_rank() == 0

def get_world_size():
    return torch.distributed.get_world_size() if torch.distributed.is_initialized() else int(os.environ.get('WORLD_SIZE', '1'))

def print0(*args, **kwargs):
    if is_main_process():
        print(*args, **kwargs)

def loginfo0(*args, **kwargs):
    if is_main_process():
        logger.info(*args, **kwargs)

#----------------------------------------------------------------------------
# Torch utils.

def replace_parameters_with_buffers_(module: torch.nn.Module) -> torch.nn.Module:
    """Turns all parameters of a module into buffers."""
    modules = module.modules()
    module = next(modules)
    for name, param in OrderedDict(module.named_parameters(recurse=False)).items():
        delattr(module, name) # Unregister parameter
        module.register_buffer(name, param.data)
    for submodule in modules:
        replace_parameters_with_buffers_(submodule)
    return module

#----------------------------------------------------------------------------
