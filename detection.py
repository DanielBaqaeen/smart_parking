import os
import wget
import tensorflow as tf
from object_detection.utils import config_util
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
import cv2
import numpy as np
from matplotlib import pyplot as plt
import yaml
from numpy import zeros

#                                       GPU Check
# from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())
# print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


# os.system('{sys.executable} -m pip install wget')
# https://www.tensorflow.org/install/source_windows
# os.system('{sys.executable} -m pip install tensorflow --upgrade')
# os.system('{sys.executable} -m pip uninstall protobuf matplotlib -y')
# os.system('{sys.executable} -m pip install protobuf matplotlib==3.2')
# get_ipython().system('{sys.executable} -m pip install object_detection')
# get_ipython().system('{sys.executable} -m pip install tensorflow-gpu')
# get_ipython().system('pip list')
# os.system('{sys.executable} -m pip install tf-nightly-gpu')


detection_model = ''
model_name = 'my_ssd_mobnet'
pretrained_name = 'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8'
model_url = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz'
generate_tf_record_script = 'generate_tfrecord.py'
label_map = 'label_map.pbtxt'

paths = {
    'workspace_path': 'workspace',
    'scripts_path': 'scripts',
    'api_model_path': 'model',
    'annotation_path': os.path.join('workspace', 'annotations'),
    'images_path': 'images',
    'model_path': os.path.join('workspace', 'models'),
    'pretrained_model_path': os.path.join('workspace', 'pre_trained_models'),
    'checkpoint_path': os.path.join('workspace', 'models', model_name),
    'protoc_path': os.path.join('protoc'),
    'output_path': os.path.join('workspace', 'models', model_name, 'export'),
    'tflite_path': os.path.join('workspace', 'models', model_name, 'tfliteexport')
}

files = {
    'pipeline_conf': os.path.join('workspace', 'models', model_name, 'pipeline.config'),
    'tf_record_script': os.path.join(paths['scripts_path'], generate_tf_record_script),
    'labelMap': os.path.join(paths['annotation_path'], label_map)
}

for path in paths.values():
    if not os.path.exists(path):
        if os.name == 'posix':
            os.system('mkdir -p ' + path)
        if os.name == 'nt':
            os.system('mkdir ' + path)


def install_object_detection():
    object_detect_path = os.path.join(paths['api_model_path'], 'research', 'object_detection')
    if not os.path.exists(object_detect_path):
        os.system("git clone https://github.com/tensorflow/models " + paths['api_model_path'])

    if os.name == 'posix':
        os.system("-m apt-get install protobuf-compiler")
        os.system(
            "cd model/research && protoc object_detection/protos/*.proto --python_out =. && cp object_detection/packages/tf2/setup.py. && python -m pip install .")

    if os.name == 'nt':
        url = "https://github.com/protocolbuffers/protobuf/releases/download/v3.15.6/protoc-3.15.6-win64.zip"
        wget.download(url)
        os.system("move protoc-3.15.6-win64.zip " + paths['protoc_path'])
        os.system("cd " + paths['protoc_path'] + " && tar -xf protoc-3.15.6-win64.zip")
        os.environ['PATH'] += os.pathsep + os.path.abspath(os.path.join(paths['protoc_path'], 'bin'))
        os.system(
            'cd model/research && protoc object_detection/protos/*.proto --python_out=. && copy object_detection\\\\packages\\\\tf2\\\\setup.py setup.py && python setup.py build && python setup.py install')
        os.system('cd model/research/slim && pip install -e . ')

    verf_script = os.path.join(paths['api_model_path'], 'research', 'object_detection', 'builders',
                               'model_builder_tf2_test.py')
    # Verify Installation
    os.system('python ' + verf_script)

    if os.name == 'posix':
        os.system("wget " + model_url)
        os.system("mv " + model_url + '.tar.gz' + paths['pretrained_model_path'])
        os.system("cd " + paths['pretrained_model_path'] + " && tar -zxvf " + pretrained_name + ".tar.gz")
    if os.name == 'nt':
        wget.download(model_url)
        os.system("move " + pretrained_name + ".tar.gz " + paths['pretrained_model_path'])
        os.system("cd " + paths['pretrained_model_path'] + " && tar -zxvf " + pretrained_name + ".tar.gz")


def label():
    labels = [{'name': 'car', 'id': 1}, {'name': 'big_object', 'id': 2}]

    with open(files['labelMap'], 'w') as f:
        for label in labels:
            f.write('item { \n')
            f.write('\tname:\'{}\'\n'.format(label['name']))
            f.write('\tid:{}\n'.format(label['id']))
            f.write('}\n')

    if not os.path.exists(files['tf_record_script']):
        os.system("git clone https://github.com/nicknochnack/GenerateTFRecord " + paths['scripts_path'])

    os.system(
        "python " + files['tf_record_script'] + " -x " + os.path.join(paths['images_path'], 'train') + " -l " + files[
            'labelMap'] + " -o " + os.path.join(paths['annotation_path'], 'train.record'))
    os.system(
        "python " + files['tf_record_script'] + " -x " + os.path.join(paths['images_path'], 'test') + " -l " + files[
            'labelMap'] + " -o " + os.path.join(paths['annotation_path'], 'test.record'))

    if os.name == 'posix':
        os.system(
            "cp " + os.path.join(paths['pretrained_model_path'], pretrained_name,
                                 'pipeline.config') + " " + os.path.join(
                paths['checkpoint_path']))
    if os.name == 'nt':
        os.system(
            "copy " + os.path.join(paths['pretrained_model_path'], pretrained_name,
                                   'pipeline.config') + " " + os.path.join(
                paths['checkpoint_path']))

    config = config_util.get_configs_from_pipeline_file(files['pipeline_conf'])

    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
    with tf.io.gfile.GFile(files['pipeline_conf'], "r") as f:
        proto_str = f.read()
        text_format.Merge(proto_str, pipeline_config)

    pipeline_config.model.ssd.num_classes = len(labels)
    pipeline_config.train_config.batch_size = 4
    pipeline_config.train_config.fine_tune_checkpoint = os.path.join(paths['pretrained_model_path'], pretrained_name,
                                                                     'checkpoint', 'ckpt-0')
    pipeline_config.train_config.fine_tune_checkpoint_type = "detection"
    pipeline_config.train_input_reader.label_map_path = files['labelMap']
    pipeline_config.train_input_reader.tf_record_input_reader.input_path[:] = [
        os.path.join(paths['annotation_path'], 'train.record')]
    pipeline_config.eval_input_reader[0].label_map_path = files['labelMap']
    pipeline_config.eval_input_reader[0].tf_record_input_reader.input_path[:] = [
        os.path.join(paths['annotation_path'], 'test.record')]

    config_text = text_format.MessageToString(pipeline_config)
    with tf.io.gfile.GFile(files['pipeline_conf'], "wb") as f:
        f.write(config_text)


def train():
    TRAINING_SCRIPT = os.path.join(paths['api_model_path'], 'research', 'object_detection', 'model_main_tf2.py')

    command = "python {} --model_dir={} --pipeline_config_path={} --num_train_steps=3000".format(TRAINING_SCRIPT, paths[
        'checkpoint_path'], files['pipeline_conf'])
    print(command)
    # os.system(command)


def evaluate_model():
    delete_this = 1
    TRAINING_SCRIPT = os.path.join(paths['api_model_path'], 'research', 'object_detection', 'model_main_tf2.py')
    command = "python {} --model_dir={} --pipeline_config_path={} --checkpoint_dir={}".format(TRAINING_SCRIPT,
                                                                                              paths['checkpoint_path'],
                                                                                              files['pipeline_conf'],
                                                                                              paths['checkpoint_path'])
    print(command)
    # os.system('{command}')


def load_ckpt():
    configs = config_util.get_configs_from_pipeline_file(files['pipeline_conf'])
    global detection_model
    detection_model = model_builder.build(model_config=configs['model'], is_training=False)

    ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
    ckpt.restore(os.path.join(paths['checkpoint_path'], 'ckpt-4')).expect_partial()


@tf.function
def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections


def detect():
    global category_index
    category_index = label_map_util.create_category_index_from_labelmap(files['labelMap'])
    IMAGE_PATH = os.path.join(paths['images_path'], 'test', 'WIN_20220411_15_03_21_Pro.jpg')

    img = cv2.imread(IMAGE_PATH)
    image_np = np.array(img)
    global num_cars
    global num_big_object
    num_cars = 0
    num_big_object = 0

    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    detections = detect_fn(input_tensor)

    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    label_id_offset = 1
    global image_np_with_detections
    image_np_with_detections = image_np.copy()

    viz_utils.visualize_boxes_and_labels_on_image_array(
        image_np_with_detections,
        detections['detection_boxes'],
        detections['detection_classes'] + label_id_offset,
        detections['detection_scores'],
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=7,
        min_score_thresh=.8,
        agnostic_mode=False)

    # COOOORDINATESS:::::

    # This is the way I'm getting my coordinates
    global boxes
    boxes = detections['detection_boxes']
    max_boxes_to_draw = boxes.shape[0]
    scores = detections['detection_scores']
    min_score_thresh = .7
    for i in range(min(max_boxes_to_draw, boxes.shape[0])):
        if scores is None or scores[i] > min_score_thresh:
            num_cars = num_cars + 1
            print("This box is gonna get used", boxes[i])


def plot_figure():
    plt.figure(figsize=(100, 50))
    plt.imshow(cv2.cvtColor(image_np_with_detections, cv2.COLOR_BGR2RGB))
    plt.savefig('prediction.png')
    # plt.show()


def coord_checker():
    num_of_spots = 7
    global ids
    ids = zeros([num_of_spots, 2], int)
    global parking_cord
    parking_cord = zeros([num_of_spots, 4], float)
    # Load the YAML file

    with open('coordinates_1.yml') as fh:

        # Load YAML data from the file

        read_data = yaml.load(fh, Loader=yaml.FullLoader)

        # Iterate the loop to read YAML data

        for i in range(0, len(read_data)):
            for key, value in read_data[i].items():
                if key == "id":
                    ids[i][0] = value
                    ids[i][1] = 0
                if key == "Ymin":
                    parking_cord[i][0] = value
                if key == "Xmin":
                    parking_cord[i][1] = value
                if key == "Ymax":
                    parking_cord[i][2] = value
                if key == "Xmax":
                    parking_cord[i][3] = value


def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    if interArea == 0:
        return 0
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


def update_spot(img):
    num_of_spots = 7
    for i in range(0, num_of_spots):
        boxA = parking_cord[i]
        for j in range(0, num_cars):
            boxB = boxes[j]
            IOUU = bb_intersection_over_union(boxA, boxB)
            if IOUU > 0.1:
                ids[i][1] = 1
                break
            else:
                ids[i][1] = 0

    height, width, c = img.shape
    for i in range(0, num_of_spots):
        a = int(parking_cord[i][0] * height)
        b = int(parking_cord[i][1] * width)
        c = int(parking_cord[i][2] * height)
        d = int(parking_cord[i][3] * width)

        if ids[i][1] == 1:
            cv2.rectangle(img, (b, a), (d, c), (0, 0, 255), 2)
        else:
            cv2.rectangle(img, (b, a), (d, c), (0, 255, 0), 2)


def real_time_detection():
    global category_index
    category_index = label_map_util.create_category_index_from_labelmap(files['labelMap'])
    cap = cv2.VideoCapture(0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    global num_cars
    global num_big_object

    while cap.isOpened():
        ret, frame = cap.read()
        image_np = np.array(frame)
        num_cars = 0
        num_big_object = 0

        input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
        detections = detect_fn(input_tensor)

        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy()
                      for key, value in detections.items()}
        detections['num_detections'] = num_detections

        # detection_classes should be ints.
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

        label_id_offset = 1
        image_np_with_detections = image_np.copy()

        viz_utils.visualize_boxes_and_labels_on_image_array(
            image_np_with_detections,

            detections['detection_boxes'],
            detections['detection_classes'] + label_id_offset,
            detections['detection_scores'],
            category_index,
            # remove scores, labels arguments to see the labels and scores on the image
            # skip_boxes=True,
            # skip_scores=True,
            # skip_labels=True,
            use_normalized_coordinates=True,
            max_boxes_to_draw=5,
            min_score_thresh=.7,
            agnostic_mode=False)

        global boxes
        boxes = detections['detection_boxes']
        max_boxes_to_draw = boxes.shape[0]
        scores = detections['detection_scores']
        min_score_thresh = .7
        for i in range(min(max_boxes_to_draw, boxes.shape[0])):
            if scores is None or scores[i] > min_score_thresh:
                num_cars = num_cars + 1

        update_spot(image_np_with_detections)

        cv2.imshow('object detection', image_np_with_detections)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break


def draw():
    img = cv2.imread('C:/Users/danie/Desktop/Senior_Spring_2022/Capstone/ML/smart_parking/parking_lot.jpg')
    height, width, c = img.shape
    a = int(parking_cord[0][0] * height)
    b = int(parking_cord[0][1] * width)
    c = int(parking_cord[0][2] * height)
    d = int(parking_cord[0][3] * width)

    cv2.rectangle(img, (b, a), (d, c), (0, 255, 0), 2)

    cv2.imshow('image', img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def freeze_convert():
    freezing_script = os.path.join(paths['api_model_path'], 'research', 'object_detection', 'exporter_main_v2.py ')
    command = "python {} --input_type=image_tensor --pipeline_config_path={} --trained_checkpoint_dir={} --output_directory={}".format(freezing_script ,files['pipeline_conf'], paths['checkpoint_path'], paths['output_path'])
    print(command)
    tflite_script = os.path.join(paths['api_model_path'], 'research', 'object_detection', 'export_tflite_graph_tf2.py ')
    command = "python {} --pipeline_config_path={} --trained_checkpoint_dir={} --output_directory={}".format(tflite_script ,files['pipeline_conf'], paths['checkpoint_path'], paths['tflite_path'])
    print(command)
    frozen_tflite = os.path.join(paths['tflite_path'], 'saved_model')
    tflite_model = os.path.join(paths['tflite_path'], 'saved_model', 'detect.tflite')
    command = "tflite_convert \
    --saved_model_dir={} \
    --output_file={} \
    --input_shapes=1,300,300,3 \
    --input_arrays=normalized_input_image_tensor \
    --output_arrays='TFLite_Detection_PostProcess','TFLite_Detection_PostProcess:1','TFLite_Detection_PostProcess:2','TFLite_Detection_PostProcess:3' \
    --inference_type=FLOAT \
    --allow_custom_ops".format(frozen_tflite, tflite_model, )
    print(command)

# Calling the functions:

# install_object_detection()
label()
#train()
#evaluate_model()
load_ckpt()
#detect()
#plot_figure()
#coord_checker()
# update_spot()
#real_time_detection()
# draw()
freeze_convert()
