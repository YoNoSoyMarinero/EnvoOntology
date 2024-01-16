import os
from PIL import Image
import PIL.Image, PIL.ImageFont, PIL.ImageDraw
import zipfile
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
import seaborn as sns

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


class_names = ["tree canopy", "water", "rock", "dirt", "sand", "waterlilly", "none", "swamp", "grass"]

# load the ground truth images and segmentation masks
# image_path = 'C:\\Users\\z0224841\\PycharmProjects\\SemWeb\\test_data\\test_img\\Image_C3_3072_10752.png'

def map_filename_to_image_and_mask(t_filename, height=512, width=512):

    '''
    Preprocesses the image by:
      * resizing the input image
      * normalizing the input image pixels

    Args:
      t_filename (string) -- path to the raw input image
      height (int) -- height in pixels to resize to
      width (int) -- width in pixels to resize to

    Returns:
      image (tensor) -- preprocessed image
    '''

    # Convert image file to tensor
    img_raw = tf.io.read_file(t_filename)
    image = tf.image.decode_png(img_raw)

    # Resize image
    image = tf.image.resize(image, (height, width,))
    image = tf.reshape(image, (height, width, 3,))

    # Normalize pixels in the input image
    image = image/127.5
    image -= 1

    return image

BATCH_SIZE = 16

def get_test_data(image_path):
    '''
    Prepares image tensor for further operation.

    Args:
      image_path (string) -- path to image file in the test set

    Returns:
      tf Dataset containing the preprocessed test set
    '''
    test_dataset = tf.data.Dataset.from_tensors(image_path)
    test_dataset = test_dataset.map(map_filename_to_image_and_mask)
    test_dataset = test_dataset.batch(1)

    return test_dataset

# test_dataset = get_test_data(image_path)

def block(x, n_convs, filters, kernel_size, activation, pool_size, pool_stride, block_name):
    '''
    Defines a block in the VGG network.

    Args:
      x (tensor) -- input image
      n_convs (int) -- number of convolution layers to append
      filters (int) -- number of filters for the convolution layers
      activation (string or object) -- activation to use in the convolution
      pool_size (int) -- size of the pooling layer
      pool_stride (int) -- stride of the pooling layer
      block_name (string) -- name of the block

    Returns:
      tensor containing the max-pooled output of the convolutions
    '''

    for i in range(n_convs):
        x = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, activation=activation, padding='same',
                                   name="{}_conv{}".format(block_name, i + 1))(x)

    x = tf.keras.layers.MaxPooling2D(pool_size=pool_size, strides=pool_stride,
                                     name="{}_pool{}".format(block_name, i + 1))(x)

    return x

vgg_weights_path = 'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'

def VGG_16(image_input):
    '''
    This function defines the VGG encoder.

    Args:
      image_input (tensor) - batch of images

    Returns:
      tuple of tensors - output of all encoder blocks plus the final convolution layer
    '''

    # create 5 blocks with increasing filters at each stage.
    # you will save the output of each block (i.e. p1, p2, p3, p4, p5). "p" stands for the pooling layer.
    x = block(image_input, n_convs=2, filters=64, kernel_size=(3, 3), activation='relu', pool_size=(2, 2),
              pool_stride=(2, 2), block_name='block1')
    p1 = x

    x = block(x, n_convs=2, filters=128, kernel_size=(3, 3), activation='relu', pool_size=(2, 2), pool_stride=(2, 2),
              block_name='block2')
    p2 = x

    x = block(x, n_convs=3, filters=256, kernel_size=(3, 3), activation='relu', pool_size=(2, 2), pool_stride=(2, 2),
              block_name='block3')
    p3 = x

    x = block(x, n_convs=3, filters=512, kernel_size=(3, 3), activation='relu', pool_size=(2, 2), pool_stride=(2, 2),
              block_name='block4')
    p4 = x

    x = block(x, n_convs=3, filters=512, kernel_size=(3, 3), activation='relu', pool_size=(2, 2), pool_stride=(2, 2),
              block_name='block5')
    p5 = x

    # create the vgg model
    vgg = tf.keras.Model(image_input, p5)

    # load the pretrained weights you downloaded earlier
    vgg.load_weights(vgg_weights_path)

    # number of filters for the output convolutional layers
    n = 4096

    # our input images are 224x224 pixels so they will be downsampled to 7x7 after the pooling layers above.
    # we can extract more features by chaining two more convolution layers.
    c6 = tf.keras.layers.Conv2D(n, (7, 7), activation='relu', padding='same', name="conv6")(p5)
    c7 = tf.keras.layers.Conv2D(n, (1, 1), activation='relu', padding='same', name="conv7")(c6)

    # return the outputs at each stage. you will only need two of these in this particular project
    # but we included it all in case you want to experiment with other types of decoders.
    return (p1, p2, p3, p4, c7)

def fcn8_decoder(convs, n_classes):
    '''
    Defines the FCN 8 decoder.

    Args:
      convs (tuple of tensors) - output of the encoder network
      n_classes (int) - number of classes

    Returns:
      tensor with shape (height, width, n_classes) containing class probabilities
    '''

    # unpack the output of the encoder
    f1, f2, f3, f4, f5 = convs

    # upsample the output of the encoder then crop extra pixels that were introduced
    o = tf.keras.layers.Conv2DTranspose(n_classes, kernel_size=(4, 4), strides=(2, 2), use_bias=False)(f5)
    o = tf.keras.layers.Cropping2D(cropping=(1, 1))(o)

    # load the pool 4 prediction and do a 1x1 convolution to reshape it to the same shape of `o` above
    o2 = f4
    o2 = (tf.keras.layers.Conv2D(n_classes, (1, 1), activation='relu', padding='same'))(o2)

    # add the results of the upsampling and pool 4 prediction
    o = tf.keras.layers.Add()([o, o2])

    # upsample the resulting tensor of the operation you just did
    o = (tf.keras.layers.Conv2DTranspose(n_classes, kernel_size=(4, 4), strides=(2, 2), use_bias=False))(o)
    o = tf.keras.layers.Cropping2D(cropping=(1, 1))(o)

    # load the pool 3 prediction and do a 1x1 convolution to reshape it to the same shape of `o` above
    o2 = f3
    o2 = (tf.keras.layers.Conv2D(n_classes, (1, 1), activation='relu', padding='same'))(o2)

    # add the results of the upsampling and pool 3 prediction
    o = tf.keras.layers.Add()([o, o2])

    # upsample up to the size of the original image
    o = tf.keras.layers.Conv2DTranspose(n_classes, kernel_size=(8, 8), strides=(8, 8), use_bias=False)(o)

    # append a softmax to get the class probabilities
    o = (tf.keras.layers.Activation('softmax'))(o)

    return o

def segmentation_model():
    '''
    Defines the final segmentation model by chaining together the encoder and decoder.

    Returns:
      keras Model that connects the encoder and decoder networks of the segmentation model
    '''

    inputs = tf.keras.layers.Input(shape=(512, 512, 3,))
    convs = VGG_16(image_input=inputs)
    outputs = fcn8_decoder(convs, 9)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    return model

model = segmentation_model()
model.load_weights('model_third_parameteres.h5')

colors = sns.color_palette(None, len(class_names))
for class_name, color in zip(class_names, colors):
    print(f'{class_name} -- {color}')

def give_color_to_annotation(annotation):
    '''
    Converts a 2-D annotation to a numpy array with shape (height, width, 3) where
    the third axis represents the color channel. The label values are multiplied by
    255 and placed in this axis to give color to the annotation

    Args:
      annotation (numpy array) - label map array

    Returns:
      the annotation array with an additional color channel/axis
    '''
    seg_img = np.zeros((annotation.shape[0], annotation.shape[1], 3)).astype('float')

    for c in range(len(class_names)):
        segc = (annotation == c)
        seg_img[:, :, 0] += segc * (colors[c][0] * 255.0)
        seg_img[:, :, 1] += segc * (colors[c][1] * 255.0)
        seg_img[:, :, 2] += segc * (colors[c][2] * 255.0)

    return seg_img

def show_predictions(image, labelmap, titles):
    '''
    Displays the image with the predicted label map

    Args:
      image (numpy array) -- the input image
      labelmap (numpy array) -- contains the predicted label map
      titles (list of strings) -- display headings for the images to be displayed
    '''

    pred_img = give_color_to_annotation(labelmap)

    image = image + 1
    image = image * 127.5
    images = np.uint8([image, pred_img])

    plt.figure(figsize=(15, 4))

    for idx, im in enumerate(images):
        plt.subplot(1, 2, idx + 1)
        # if idx == 1:
        #     plt.xlabel(display_string)
        plt.xticks([])
        plt.yticks([])
        plt.title(titles[idx], fontsize=12)
        plt.imshow(im)

    plt.show()

def get_images_and_segments_test_arrays(test_dataset):
    '''
    Gets a subsample of the test set

    Returns:
      Test set containing ground truth image
    '''
    y_true_image = []

    ds = test_dataset.unbatch()
    ds = ds.batch(1)

    for image in ds.take(1):
        y_true_image = image

    return y_true_image

def get_dataset_slice_paths(image_dir):
    '''
    generates the lists of image and label map paths

    Args:
      image_dir (string) -- path to the input images directory
      label_map_dir (string) -- path to the label map directory

    Returns:
      image_paths (list of strings) -- paths to each image file
      label_map_paths (list of strings) -- paths to each label map
    '''
    image_file_list = os.listdir(image_dir)
    image_paths = [os.path.join(image_dir, fname) for fname in image_file_list]

    return image_paths

# load the ground truth images and segmentation masks
# y_true_image = get_images_and_segments_test_arrays()

validation_count = 89
validation_steps = validation_count//BATCH_SIZE

# get the model prediction
# results = model.predict(test_dataset)

# for each pixel, get the slice number which has the highest probability
# results = np.argmax(results, axis=3)

# show_predictions(y_true_image[0], results[0], ["Image", "Predicted Mask"])
