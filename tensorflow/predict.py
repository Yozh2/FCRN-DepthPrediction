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

def load_image(image_path, height=1024, width=768):
    '''Load and preprocess image from `image_path` given'''
    
    # Load original image
    image_cv2 = cv2.imread(image_path)
    print(f'Original image loaded with shape: {image_cv2.shape}')
    
    # Resize image to the requested shapes 
    image_cv2 = cv2.resize(image_cv2, (width, height), interpolation = cv2.INTER_AREA)
    print(f'Resized image shape: {image_cv2.shape}')

    # Preprocess image to use it with tensorflow
    image_cv2 = image_cv2.astype('float32')
    image_cv2 = np.expand_dims(image_cv2, axis=0)
    print(f'Postprocessed image shape: {image_cv2.shape}')
    return image_cv2
    
def save_depth(depth, path):
    '''Save depth as numpy pickle and as image'''
    fig = plt.figure()
    ii = plt.imshow(depth[0,:,:,0], interpolation='nearest')
    fig.colorbar(ii)
    # plt.show()
    plt.savefig('%s.png' % path)
    np.save('%s.npy' % path, depth)       
    
def predict_dir(model_path, dir_path):
    img_fnames = os.listdir(dir_path)
    
    for path in img_fnames:
        image_path = os.path.join(dir_path, path)
        image = load_image(image_path)
        pred = predict(model_path, image)
        save_depth(pred, image_path + '_depth')
    
def predict(model_data_path, image_cv2, channels=3, batch_size=1):
    
    dims = image_cv2.shape
    
    # Create a placeholder for the input image
    input_node = tf.placeholder(tf.float32, shape=(None, dims[1], dims[2], channels))

    # Construct the network
    net = models.ResNet50UpProj({'data': input_node}, batch_size, 1, False)
        
    with tf.Session() as sess:

        # Load the converted parameters
        print('Loading the model')

        # Use to load from ckpt file
        saver = tf.train.Saver()     
        saver.restore(sess, model_data_path)

        # Use to load from npy file
        #net.load(model_data_path, sess) 

        # Evalute the network for the given image
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
    
                            
                            
    # make predictions
    if args.dir:
        predict_dir(args.model_path, args.image_path)
    else:
        image = load_image(args.image_path)
        pred = predict(args.model_path, image)
        save_depth(pred, args.image_path + '_depth')

    os._exit(0)

if __name__ == '__main__':
    main()    
