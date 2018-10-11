import argparse
import os
import numpy as np
import tensorflow as tf

# Disable matplotlib drawing frontend because it is not available on the server
import matplotlib
matplotlib.use('agg')

import matplotlib.pyplot as plt
import cv2
import models

class FCRNDepthPredictor:
    def __init__(self, model_path=None, image_path=None, dir_path=None,
                 channels=3, batch_size=1, verbose=False):
        '''Construct TensorFlow graph to predict depth in the future.

        Args:
            model_path (str):
                the path to pretrained model weights

            image_path (str):
                the path to the image the depth need to be estimated to

            dir_path (str):
                the path to the directory containing images that are to
                be predicted depth to

            channels (int):
                number of image channels, such as RGB (channels == 3)

            batch_size (int):
                the size of the batch size for ResNet

            verbose (bool):
                toggle print debug information to the console
        '''
        self.verbose = verbose
        self.model_path = model_path
        self.image_path = image_path
        self.dir_path = dir_path

        # Remember tensorflow session
        self.sess = sess
        self.batch_size = batch_size
        self.channels = channels
        self.saver = tf.train.Saver()

        # Create a placeholder for the input image
        self.input_node = tf.placeholder(tf.float32, shape=(None, None, None, channels))
        if self.verbose:
            print('-> Input placeholder created...')

        # Construct the network
        self.net = models.ResNet50UpProj({'data': self.input_node},
                                         self.batch_size, 1, False)
        if self.verbose:
            print('-> ResNet50 initialized...')


    def load_weights(self, sess=None):
        '''Load model's pretrained weights'''
        # Use to load from ckpt file
        self.saver.restore(sess, self.model_path)

        # Use to load from npy file
        #net.load(model_data_path, sess)


    def load_image(self, image_path, height=1024, width=768):
        '''Load and preprocess image from `image_path` given.'''

        # Load original image
        if not os.path.exists(image_path):
            raise FileNotFoundError
        else:
            image_cv2 = cv2.imread(image_path)
        if self.verbose:
            print(f'Original image loaded with shape: {image_cv2.shape}')

        # Resize image to the requested shapes
        if height is None:
            height = image_cv2.shape[0]
        if width is None:
            width = image_cv2.shape[1]
        image_cv2 = cv2.resize(image_cv2, (width, height),
            interpolation = cv2.INTER_AREA)
        if self.verbose:
            print(f'Resized image shape: {image_cv2.shape}')

        # Preprocess image to use it with tensorflow
        image_cv2 = image_cv2.astype('float32')
        image_cv2 = np.expand_dims(image_cv2, axis=0)
        if self.verbose:
            print(f'Postprocessed image shape: {image_cv2.shape}')
        return image_cv2


    def save_depth(self, depth, path):
        '''Save depth as numpy pickle and as image'''
        fig = plt.figure()
        ii = plt.imshow(depth[0,:,:,0], interpolation='nearest')
        fig.colorbar(ii)
        plt.savefig('%s.png' % path)
        np.save('%s.npy' % path, depth)


    def predict_dir(self, model_path, dir_path):
        '''Predict depth for all images in the dir_path'''
        img_fnames = os.listdir(dir_path)

        for path in img_fnames:
            image_path = os.path.join(dir_path, path)
            image = self.load_image(image_path)
            pred = self.predict(model_path, image)
            self.save_depth(pred, image_path + '_depth')


    def predict(self, image_cv2, sess):
        '''Evalute the network for the given image'''
        pred = sess.run(net.get_output(), feed_dict={input_node: image_cv2})
        return pred


def main():

    def parse_args():
        """Parses arguments and returns args object to the main program"""
        parser = argparse.ArgumentParser()
        parser.add_argument('-m', '--model_path', default='models/NYU_FCRN.ckpt',
                            help='Converted parameters for the model')
        parser.add_argument('-i', '--image_path', default='images/image.png',
                            help='Image (or directory) to predict depth for')
        parser.add_argument('-d', '--dir', action='store_true', default=False,
                            help='Interpret image path as a directory containing images. \
                            Make prediction for every image in the directory.')
        return parser.parse_known_args()

    # parse arguments
    args, unknown = parse_args()

    fcrn = FCRNDepthPredictor(args.model_path, args.image_path, args.depth_path,
                              channels=3, batch_size=1, verbose=True)

    with tf.Session() as sess:
        fcrn.load_weights(sess)

        # make predictions
        if args.dir:
            fcrn.predict_dir(args.model_path, args.image_path)
        else:
            image = fcrn.load_image(args.image_path)
            pred = fcrn.predict(args.model_path, image)
            fcrn.save_depth(pred, args.image_path + '_depth')

    os._exit(0)

if __name__ == '__main__':
    main()
