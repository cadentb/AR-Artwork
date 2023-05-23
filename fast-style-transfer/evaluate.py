from __future__ import print_function
import sys
sys.path.insert(0, 'src')
import transform, numpy as np, vgg, pdb, os
import scipy.misc
import tensorflow as tf
from utils import save_img, get_img, exists, list_files
from argparse import ArgumentParser
from collections import defaultdict
import time
import json
import subprocess
import numpy
from moviepy.video.io.VideoFileClip import VideoFileClip
import moviepy.video.io.ffmpeg_writer as ffmpeg_writer

BATCH_SIZE = 4
DEVICE = '/gpu:0'

# input video, output video, checkpoint directory, [computation device, batch size]
# Stands for feedforward
def ffwd_video(path_in, path_out, checkpoint_dir, device_t='/gpu:0', batch_size=4):

    #The input video is loaded using the VideoFileClip class from the moviepy library. The audio argument is set to False to disable audio processing.
    video_clip = VideoFileClip(path_in, audio=False)

    # A FFMPEG_VideoWriter object is created to write the output video to disk. 
    # The object is configured with the output path (path_out), the size of the video (video_clip.size), 
    #   the frame rate (video_clip.fps), and various encoding settings (codec, preset, bitrate, and audiofile). 
    # The threads and ffmpeg_params arguments are left to their default values.
    video_writer = ffmpeg_writer.FFMPEG_VideoWriter(path_out, video_clip.size, video_clip.fps, codec="libx264",
                                                    preset="medium", bitrate="2000k",
                                                    audiofile=path_in, threads=None,
                                                    ffmpeg_params=None)

    # A new TensorFlow graph is created, along with a session using the specified device (device_t)
    #  and a soft placement configuration that allows TensorFlow to fall back to CPU execution if necessary.
    g = tf.Graph()
    soft_config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
    soft_config.gpu_options.allow_growth = True

    with g.as_default(), g.device(device_t), \
            tf.compat.v1.Session(config=soft_config) as sess:
        batch_shape = (batch_size, video_clip.size[1], video_clip.size[0], 3)

        #The input image placeholder is defined using tf.placeholder, with a shape of (batch_size, height, width, channels) and a name of img_placeholder.
        img_placeholder = tf.compat.v1.placeholder(tf.float32, shape=batch_shape,
                                         name='img_placeholder')

        # The style transfer model is loaded using transform.net, 
        #   which is a function that returns a TensorFlow computation graph that performs style transfer. 
        # The input to the graph is img_placeholder, and the output is preds, which represents the stylized image.
        preds = transform.net(img_placeholder)

        # A TensorFlow Saver object is created to load the weights of the model from the checkpoint directory (checkpoint_dir). 
        # If the checkpoint directory is a directory, the latest checkpoint file is loaded. If it's a file path, the specified checkpoint file is loaded.
        saver = tf.compat.v1.train.Saver()
        if os.path.isdir(checkpoint_dir):
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                raise Exception("No checkpoint found...")
        else:
            saver.restore(sess, checkpoint_dir)

        # An empty numpy array X is created with the shape of (batch_size, height, width, channels) to store the input frames.
        X = np.zeros(batch_shape, dtype=np.float32)

        # The style_and_write function is defined to perform style transfer on a batch of frames, and write the stylized frames to the output video using video_writer.write_frame. 
        # This function takes one argument, count, which is the number of frames to process in the current batch.
        # Processes the batch of frames in X, writes the stylized frames to the output video, and resets X to an empty array.
        def style_and_write(count):
            for i in range(count, batch_size):
                X[i] = X[count - 1]  # Use last frame to fill X
            _preds = sess.run(preds, feed_dict={img_placeholder: X})
            for i in range(0, count):
                video_writer.write_frame(np.clip(_preds[i], 0, 255).astype(np.uint8))

        # The main loop begins, which iterates over each frame in the input video using video_clip.iter_frames. 
        # For each frame, the frame is stored in the X array at the current index (frame_count), and frame_count is incremented.
        frame_count = 0  # The frame count that written to X
        for frame in video_clip.iter_frames():
            X[frame_count] = frame
            frame_count += 1

            # If frame_count reaches batch_size, the style_and_write function is called with the current frame_count. 
            # This function processes the batch of frames in X, writes the stylized frames to the output video, and resets X to an empty array.
            if frame_count == batch_size:
                style_and_write(frame_count)
                frame_count = 0

        # If the end of the video is reached and frame_count is less than batch_size, the remaining frames in X are processed using style_and_write.
        if frame_count != 0:
            style_and_write(frame_count)

        # The FFMPEG_VideoWriter is closed using video_writer.close.
        video_writer.close()


# get img_shape. Stands for feedforward.
def ffwd(data_in, paths_out, checkpoint_dir, device_t='/gpu:0', batch_size=4):
    assert len(paths_out) > 0
    
    # Check whether the input data is a set of image paths or a set of image data.
    # If it is a set of paths, it loads the first image to determine the image shape. 
    # If it is a set of image data, it uses the shape of the first image in the set.
    is_paths = type(data_in[0]) == str
    if is_paths:
        assert len(data_in) == len(paths_out)
        img_shape = get_img(data_in[0]).shape
    else:
        assert data_in.size[0] == len(paths_out)
        img_shape = X[0].shape

    # Set up a TensorFlow graph and session with specified device for computation (either a GPU or CPU).
    g = tf.Graph()
    batch_size = min(len(paths_out), batch_size)
    curr_num = 0
    soft_config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
    soft_config.gpu_options.allow_growth = True
    with g.as_default(), g.device(device_t), \
            tf.compat.v1.Session(config=soft_config) as sess:
        batch_shape = (batch_size,) + img_shape

        # placeholder for the input images
        img_placeholder = tf.compat.v1.placeholder(tf.float32, shape=batch_shape,
                                         name='img_placeholder')

        preds = transform.net(img_placeholder)

        # Restore a trained model from the specified checkpoint directory using TensorFlow's Saver class.
        saver = tf.compat.v1.train.Saver()
        if os.path.isdir(checkpoint_dir):
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                raise Exception("No checkpoint found...")
        else:
            saver.restore(sess, checkpoint_dir)

        # Loop over the output paths in batches.
        num_iters = int(len(paths_out)/batch_size)
        for i in range(num_iters):

            pos = i * batch_size
            curr_batch_out = paths_out[pos:pos+batch_size]

            # Load a batch of input images.
            if is_paths:
                curr_batch_in = data_in[pos:pos+batch_size]
                X = np.zeros(batch_shape, dtype=np.float32)
                for j, path_in in enumerate(curr_batch_in):
                    img = get_img(path_in)
                    assert img.shape == img_shape, \
                        'Images have different dimensions. ' +  \
                        'Resize images or use --allow-different-dimensions.'
                    X[j] = img
            else:
                X = data_in[pos:pos+batch_size]

            # Process the batch of input images using the pre-trained model.
            _preds = sess.run(preds, feed_dict={img_placeholder:X})

            # Save the corresponding output images.
            for j, path_out in enumerate(curr_batch_out):
                save_img(path_out, _preds[j])
                
        remaining_in = data_in[num_iters*batch_size:]
        remaining_out = paths_out[num_iters*batch_size:]
    
    # If the number of output paths is not evenly divisible by the batch size, recursively call ffwd with the remaining inputs and outputs.
    if len(remaining_in) > 0:
        ffwd(remaining_in, remaining_out, checkpoint_dir, 
            device_t=device_t, batch_size=1)

# In essence, this function is a wrapper for the ffwd function that processes a single input image and saves the corresponding output image to the specified output path. 
# It is intended to be more convenient to use for processing individual images, rather than batching multiple images at once.
def ffwd_to_img(in_path, out_path, checkpoint_dir, device='/cpu:0'):
    paths_in, paths_out = [in_path], [out_path]
    ffwd(paths_in, paths_out, checkpoint_dir, batch_size=1, device_t=device)

def ffwd_different_dimensions(in_path, out_path, checkpoint_dir, 
            device_t=DEVICE, batch_size=4):
    
    # Uses defaultdict() to group together input and output paths that correspond to images with the same shape.
    in_path_of_shape = defaultdict(list)
    out_path_of_shape = defaultdict(list)

    # Categorizes the input and output paths based on the shape of the corresponding images.
    for i in range(len(in_path)):
        in_image = in_path[i]
        out_image = out_path[i]

        # obtain the shape of each input image, format it as a string
        shape = "%dx%dx%d" % get_img(in_image).shape

        # defaultdict groups images with the same shape
        in_path_of_shape[shape].append(in_image)
        out_path_of_shape[shape].append(out_image)
    
    # loops over each set of input and output paths with the same shape and calls the ffwd function to transform 
    #   the input images to output images in the style of a specified image using a pre-trained model. 
    # The output images are saved to their corresponding output paths.
    for shape in in_path_of_shape:
        print('Processing images of shape %s' % shape)
        ffwd(in_path_of_shape[shape], out_path_of_shape[shape], 
            checkpoint_dir, device_t, batch_size)

# Sets up several arguments that can be passed to the program:
def build_parser():
    parser = ArgumentParser()

    # --checkpoint
    # a required argument that specifies the path to the checkpoint directory or file from which to load the pre-trained style transfer model.
    parser.add_argument('--checkpoint', type=str,
                        dest='checkpoint_dir',
                        help='dir or .ckpt file to load checkpoint from',
                        metavar='CHECKPOINT', required=True)

    # --in-path 
    # a required argument that specifies the input image or directory of images to transform.
    parser.add_argument('--in-path', type=str,
                        dest='in_path',help='dir or file to transform',
                        metavar='IN_PATH', required=True)

    help_out = 'destination (dir or file) of transformed file or files'

    # --out-path
    # a required argument that specifies the output directory or file to which the transformed image(s) will be written.
    parser.add_argument('--out-path', type=str,
                        dest='out_path', help=help_out, metavar='OUT_PATH',
                        required=True)

    # --device
    # an optional argument that specifies the device (e.g. 'cpu:0' or 'gpu:0') to use for computation. The default is 'cpu:0'.
    parser.add_argument('--device', type=str,
                        dest='device',help='device to perform compute on',
                        metavar='DEVICE', default=DEVICE)

    # --batch-size
    # an optional argument that specifies the batch size to use for feedforwarding. The default is 4.
    parser.add_argument('--batch-size', type=int,
                        dest='batch_size',help='batch size for feedforwarding',
                        metavar='BATCH_SIZE', default=BATCH_SIZE)

    # --allow-different-dimensions
    # an optional argument that allows the program to handle images with different dimensions.
    parser.add_argument('--allow-different-dimensions', action='store_true',
                        dest='allow_different_dimensions', 
                        help='allow different image dimensions')

    return parser

# The function check_opts(opts) checks the validity of the options (or arguments) passed to the program. 
# It takes the opts object as an argument which is created by parsing the command-line arguments using the build_parser() function.
# Specifically, the function checks whether the checkpoint directory and input path exist, and if the output path is a directory
# , it also checks if the directory exists. 
# It then asserts that the batch size is greater than zero.
# If any of the checks fail, an error message is displayed, indicating the nature of the problem.
def check_opts(opts):
    exists(opts.checkpoint_dir, 'Checkpoint not found!')
    exists(opts.in_path, 'In path not found!')
    if os.path.isdir(opts.out_path):
        exists(opts.out_path, 'out dir not found!')
        assert opts.batch_size > 0

def main():

    # builds a parser object to parse the command-line arguments using the build_parser() function.
    parser = build_parser()

    # parses the arguments using parse_args() method of the parser object.
    opts = parser.parse_args()

    # check the validity of the command-line arguments. 
    # It checks if the checkpoint directory and input path exist. 
    # If the output path is a directory, it also checks if it exists and if the batch size is greater than 0.
    check_opts(opts)

    # If the input path is a file
    if not os.path.isdir(opts.in_path):

        # If the output path is a directory
        if os.path.exists(opts.out_path) and os.path.isdir(opts.out_path):

            # creates the output file in that directory with the same name as the input file
            out_path = \
                    os.path.join(opts.out_path,os.path.basename(opts.in_path))
            
        # If the output path is not a directory (aka a file), it uses it as the output file path
        else:
            out_path = opts.out_path

        # calls the ffwd_to_img() function with the appropriate parameters to perform feedforwarding.
        ffwd_to_img(opts.in_path, out_path, opts.checkpoint_dir,
                    device=opts.device)
    
    # If the input path is a directory
    else:

        # gets the list of files in that directory
        files = list_files(opts.in_path)

        # creates a list of full input file paths
        full_in = [os.path.join(opts.in_path,x) for x in files]

        # creates a list of full output file paths
        full_out = [os.path.join(opts.out_path,x) for x in files]

        # If the allow_different_dimensions flag is set, it calls the ffwd_different_dimensions() to perform feedforwarding
        if opts.allow_different_dimensions:
            ffwd_different_dimensions(full_in, full_out, opts.checkpoint_dir, 
                    device_t=opts.device, batch_size=opts.batch_size)
            
        # Otherwise, it calls ffwd() to perform feedforwarding
        else :
            ffwd(full_in, full_out, opts.checkpoint_dir, device_t=opts.device,
                    batch_size=opts.batch_size)

if __name__ == '__main__':
    main()
