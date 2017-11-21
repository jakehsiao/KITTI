import tensorflow as tf
import os
from object_detection.utils import dataset_util
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt


flags = tf.app.flags
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
FLAGS = flags.FLAGS


def create_tf_example(txt, png):
   
    img = Image.open(png)
    width, height = img.size
    with tf.gfile.GFile(png, 'rb') as fid:
        encoded_png = fid.read() # Encoded image bytes
       
    image_format = b'png' # b'jpeg' or b'png'

    xmins = [] # List of normalized left x coordinates in bounding box (1 per box)
    xmaxs = [] # List of normalized right x coordinates in bounding box
             # (1 per box)
    ymins = [] # List of normalized top y coordinates in bounding box (1 per box)
    ymaxs = [] # List of normalized bottom y coordinates in bounding box
             # (1 per box)
    classes_text = [] # List of string class name of bounding box (1 per box)
    classes = [] # List of integer class id of bounding box (1 per box)

    
    with open(txt, 'r') as handle:
        for line in handle.readlines():
            tokens = line.split(' ')
            xmins.append(float(tokens[1]))
            ymins.append(float(tokens[2]))
            xmaxs.append(float(tokens[3]))
            ymaxs.append(float(tokens[4]))
            classes_text.append(b'Car')
            classes.append(1)
    
    
                  
    tf_example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/encoded': dataset_util.bytes_feature(encoded_png),
      'image/format': dataset_util.bytes_feature(image_format),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
      'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
      'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


def main(_):
    writer = tf.python_io.TFRecordWriter(FLAGS.output_path)
    path = '/home/ypw/Desktop/KITTI/TSD_train/'
    txts = filter(lambda x : 'txt' in x, os.listdir(path))

    for txt in tqdm(txts):
        txt_abs = os.path.join(path, txt)
        png_abs = os.path.join(path, txt[:-4]+'.png')
        tf_example = create_tf_example(txt_abs, png_abs)
        writer.write(tf_example.SerializeToString())

    writer.close()


if __name__ == '__main__':
    tf.app.run()