import colorsys
import datetime
import os
from enum import Enum


class ToonstoryAI(Enum):
    BEARD = 0
    CLAB = 1
    DLIB = 2
    GLASSES = 3
    HAIR_CURL = 4
    HAIR_FRONT = 5
    HAIR_LONG = 6
    TEXT_CNN = 7
    WORD2VEC = 8


# PORT
BEARD_STYLE_CNN_PREDICTION_WEBSERVER_PORT = 7070
CLAB_PORT = 2222
DLIB_PREDICTION_WEBSERVER_PORT = 9999
GLASSESS_STYLE_CNN_PREDICTION_WEBSERVER_PORT = 7777
HAIR_CURL_CNN_PREDICTION_WEBSERVER_PORT = 5050
HAIR_FRONT_CNN_PREDICTION_WEBSERVER_PORT = 5555
HAIR_LONG_CNN_PREDICTION_WEBSERVER_PORT = 6666
TEXTCNN_PREDICTION_WEBSERVER_PORT = 9090
WORD2VEC_PORT = 8888

# PORTS = [7070, 2222, 9999, 7777, 5050, 5555, 6666, 9090, 8888]
PORTS = {
    ToonstoryAI.BEARD.name: 7070,
    ToonstoryAI.CLAB.name: 2222,
    ToonstoryAI.DLIB.name: 9999,
    ToonstoryAI.GLASSES.name: 7777,
    ToonstoryAI.HAIR_CURL.name: 5050,
    ToonstoryAI.HAIR_FRONT.name: 5555,
    ToonstoryAI.HAIR_LONG.name: 6666,
    ToonstoryAI.TEXT_CNN.name: 9090,
    ToonstoryAI.WORD2VEC.name: 8888
}
AI_PATHS = {
    ToonstoryAI.BEARD.name: "beard_cnn",
    ToonstoryAI.CLAB.name: "cnn",
    ToonstoryAI.DLIB.name: "dlib",
    ToonstoryAI.GLASSES.name: "glasses_cnn",
    ToonstoryAI.HAIR_CURL.name: "hair_curl",
    ToonstoryAI.HAIR_FRONT.name: "hair_front",
    ToonstoryAI.HAIR_LONG.name: "hair_long",
    ToonstoryAI.TEXT_CNN.name: "textcnn",
    ToonstoryAI.WORD2VEC.name: "word2vec"}

AI_IMAGE = {
    ToonstoryAI.BEARD.name: {
        "WIDTH": 87,
        "HEIGHT": 71
    },
    ToonstoryAI.CLAB.name: {
        "WIDTH": 128,
        "HEIGHT": 128
    },
    ToonstoryAI.DLIB.name: {
        "WIDTH": 64,
        "HEIGHT": 64
    },
    ToonstoryAI.GLASSES.name: {
        "WIDTH": 100,
        "HEIGHT": 54
    },
    ToonstoryAI.HAIR_CURL.name: {
        "WIDTH": 100,
        "HEIGHT": 100
    },
    ToonstoryAI.HAIR_FRONT.name: {
        "WIDTH": 70,
        "HEIGHT": 66
    },
    ToonstoryAI.HAIR_LONG.name: {
        "WIDTH": 60,
        "HEIGHT": 124
    },
    ToonstoryAI.TEXT_CNN.name: {
        "WIDTH": None,
        "HEIGHT": None

    },
    ToonstoryAI.WORD2VEC.name: {
        "WIDTH": None,
        "HEIGHT": None

    }

}

now_instance = datetime.datetime.now()
# MONGODB_ADDRESS = 'localhost'
# MONGODB_ADDRESS = 'mongodb://toonstory:PKNb9NUstCk09keuO2wsWgoQIe0PMT2iL9tWNwpH0QTY8hVywhsxGV5hGrv6gKG5VUHTha6pZOkOXxPH1QX5wA==@toonstory.documents.azure.com:10255/?ssl=true&replicaSet=globaldb'
MONGODB_ADDRESS = 'mongodb://toonsquare:040201m~!@mongodb.toonsquare.co:27017/admin'
# for default cnn meta data for clab far
TRAIN_TIMES = 500
TRAIN_BATCH_SIZE = 200

TEST_SIZE = 200
TEST_BATCH_SIZE = 100

# for hair length
TRAIN_TIMES_HAIR_LONG = 200
TRAIN_BATCH_SIZE_HAIR_LONG = 100

TEST_SIZE_HAIR_LONG = 100
TEST_BATCH_SIZE_HAIR_LONG = 100

# for hair front
TRAIN_TIMES_HAIR_FRONT = 200
TRAIN_BATCH_SIZE_HAIR_FRONT = 200

TEST_SIZE_HAIR_FRONT = 70
TEST_BATCH_SIZE_HAIR_FRONT = 50

# for hair long
TRAIN_TIMES_HAIR_LONG = 200
TRAIN_BATCH_SIZE_HAIR_LONG = 50

TEST_SIZE_HAIR_LONG = 70
TEST_BATCH_SIZE_HAIR_LONG = 50

# for hair curl
TRAIN_TIMES_HAIR_CURL = 100
TRAIN_BATCH_SIZE_HAIR_CURL = 200

TEST_SIZE_HAIR_CURL = 100
TEST_BATCH_SIZE_HAIR_CURL = 100

# for beard cnn
TRAIN_TIMES_BEARD = 700
TRAIN_BATCH_SIZE_BEARD = 50

TEST_SIZE_BEARD = 100
TEST_BATCH_SIZE_BEARD = 100

# for glasses cnn
TRAIN_TIMES_GLASSES = 700
TRAIN_BATCH_SIZE_GLASSES = 40

TEST_SIZE_GLASSES = 70
TEST_BATCH_SIZE_GLASSES = 50

MAX_PATCH = 200
MAX_MINOR = 100
MAX_MAJOR = 50

BEARD_WIDTH = 87
BEARD_HEIGHT = 71

CLAB_WIDTH = 128
CLAB_HEIGHT = 128

JAW_WIDTH = 64
JAW_HEIGHT = 64

GLASSES_WIDTH = 100
GLASSES_HEIGHT = 54

HAIR_CURL_WIDTH = 100
HAIR_CURL_HEIGHT = 100

HAIR_FRONT_WIDTH = 70
HAIR_FRONT_HEIGHT = 66

HAIR_LONG_WIDTH = 60
HAIR_LONG_HEIGHT = 124

TIME_TAG = now_instance.strftime('%Y%m%d%H%M%S.')

TRAIN_FILENAME = 'story_train.txt'
TRAIN_DATA_FILENAME = TRAIN_FILENAME + '.data'
TRAIN_VOCAB_FILENAME = TRAIN_FILENAME + '.vocab'

TEST_FILENAME = 'story_test.txt'
TEST_DATA_FILENAME = TEST_FILENAME + '.data'
TEST_VOCAB_FILENAME = TEST_FILENAME + '.vocab'

CNN_MODEL_LATEST_VERSION_NAME = 'model/max_accuracy.ckpt'
TEXTCNN_MODEL_LATEST_VERSION_NAME = 'model/max_accuracy.ckpt'
WORD2VEC_LATEST_VERSION_NAME = 'model/word2vec.model'
DOC2VEC_LATEST_VERSION_NAME = 'model/doc2vec.model'
VERSION_FILE = 'version.json'

SUPVERVIDED_DATA_PATH = 'supervised_data'
AZURE_SUPVERVIDED_DATA_PATH = 'azure_supervised_data'
PREDICTION_TEST_PATH = 'prediction_test'
PREDICTION_REQ_PATH = 'prediction_request'
STATIC_URL_PREFIX = '/images/'

# azure storage file service access account
STORAGE_ACCOUNT_NAME = 'itself'
STORAGE_ACCOUNT_KEY = 'nLwLui04/u7cL8suwhFQxDw5uc4KBEB4eOLG5rnJceAl4pMIXH5r1OWkG3ApoQM0Swhah83B2xNTzQF7ShQhDQ=='
STORAGE_NAME = 'ai-model'

SUPERVISED_STORAGE_NAME = 'supervised-data'
SUPERVISED_STORAGE_NAME_BEARD = 'beard-supervised-data'
SUPERVISED_STORAGE_NAME_GLASSES = 'glasses-supervised-data'
SUPERVISED_STORAGE_NAME_HAIR_LONG = 'hair-long-supervised-data'
SUPERVISED_STORAGE_NAME_HAIR_FRONT = 'hair-front-supervised-data'
SUPERVISED_STORAGE_NAME_HAIR_CURL = 'hair-curl-supervised-data'

SUPERVISED_STORAGE_NAME = {
    ToonstoryAI.BEARD.name: 'beard-supervised-data',
    ToonstoryAI.CLAB.name: 'supervised-data',
    ToonstoryAI.GLASSES.name: 'glasses-supervised-data',
    ToonstoryAI.HAIR_CURL.name: 'hair-curl-supervised-data',
    ToonstoryAI.HAIR_FRONT.name: 'hair-front-supervised-data',
    ToonstoryAI.HAIR_LONG.name: 'hair-long-supervised-data',
    ToonstoryAI.TEXT_CNN.name: 'mongodb',
    ToonstoryAI.WORD2VEC.name: 'mongodb'
}

STORAGE_TEXTCNN_DIR = 'textcnn'
STORAGE_CNN_DIR = 'cnn'
STORAGE_CNN_DIR_BEARD = 'beard_cnn'
STORAGE_CNN_DIR_GLASSES = 'glasses_cnn'
STORAGE_CNN_DIR_HAIR_LONG = 'hair_long_cnn'
STORAGE_CNN_DIR_HAIR_FRONT = 'hair_front_cnn'
STORAGE_CNN_DIR_HAIR_CURL = 'hair_curl_cnn'
STORAGE_WORD2VEC = "word2vec"
STORAGE_BERT_SENTIMENT_DIR = "bert_sentiment"

STORAGE_CNN_DIR = {
    ToonstoryAI.BEARD.name: 'beard_cnn',
    ToonstoryAI.CLAB.name: 'cnn',
    ToonstoryAI.GLASSES.name: 'glasses_cnn',
    ToonstoryAI.HAIR_CURL.name: 'hair_curl_cnn',
    ToonstoryAI.HAIR_FRONT.name: 'hair_front_cnn',
    ToonstoryAI.HAIR_LONG.name: 'hair_long_cnn',
    ToonstoryAI.TEXT_CNN.name: 'textcnn',
    ToonstoryAI.WORD2VEC.name: 'word2vec'
}

# model deploy
MODEL_ZIP_NAME = 'model.zip'
MODEL_DIR_PATH = 'model'

#bert_config.json 
BERT_CONFIG_NAME = 'bert_config.json'

PRODUCTION = 'production'
# LOG
LOG_PATH_NAME = './logs/nn_logs'

# textcnn input word length
TEXTCNN_INPUT_WORD_LENGTH = 30

# slack token
SLACK_TOKEN = 'xoxp-727903327684-728365517520-895625303397-a786f9e885ac81c31615597de0f16fed'

# dlib face landmark data
SHAPE_PREDICTOR_68_FACE_LANDMARK_DATA = 'shape_predictor_68_face_landmarks.dat'

PROTOCOL = "http://"
DOMAIN = "www.toonstory.co"
DEV_DOMAIN = "localhost"
NODE_IMAGE_PATH = "/home/toonstory/deploy/toonstory_web/app/public/images"

BASE64_PREFIX = "data:image/jpeg;base64,"

# channel

HAIR_LONG_CHANNEL = 3

# face aligner default with height
DESIRE_FACE_WIDTH = 128
DESIRE_FACE_HEIGHT = 230

DESIRE_CURL_WIDTH = 800
DESIRE_CURL_HEIGHT = 1000

FACE_ALIGNE_DESIRED_LEFT_EYE = {
    ToonstoryAI.BEARD.name: (0.35, 0.35),
    ToonstoryAI.CLAB.name: (0.35, 0.35),
    ToonstoryAI.DLIB.name: (0.35, 0.35),
    ToonstoryAI.GLASSES.name: (0.35, 0.35),
    ToonstoryAI.HAIR_CURL.name: (0.35, 0.5),
    ToonstoryAI.HAIR_FRONT.name: (0.35, 0.35),
    ToonstoryAI.HAIR_LONG.name: (0.35, 0.35),
    ToonstoryAI.TEXT_CNN.name: (0.35, 0.35),
    ToonstoryAI.WORD2VEC.name: (0.35, 0.35),
}

FACE_ALIGNE_FACE_DESIRE_SIZE = {
    ToonstoryAI.BEARD.name: {
        "WIDTH": 128,
        "HEIGHT": 230  # BEARD
    },
    ToonstoryAI.CLAB.name: {
        "WIDTH": 128,
        "HEIGHT": 230  # CLAB
    },
    ToonstoryAI.DLIB.name: {
        "WIDTH": 128,
        "HEIGHT": 230  # DLIB
    },
    ToonstoryAI.GLASSES.name: {
        "WIDTH": 128,
        "HEIGHT": 230  # GLASSES
    },
    ToonstoryAI.HAIR_CURL.name: {
        "WIDTH": 800,
        "HEIGHT": 1000  # HAIR_CURL
    },
    ToonstoryAI.HAIR_FRONT.name: {
        "WIDTH": 128,
        "HEIGHT": 230  # HAIR_FRONT
    },
    ToonstoryAI.HAIR_LONG.name: {
        "WIDTH": 128,
        "HEIGHT": 230  # HAIR_LONG
    },
    ToonstoryAI.TEXT_CNN.name: {
        "WIDTH": None,
        "HEIGHT": None  # TEST_CNN

    },
    ToonstoryAI.WORD2VEC.name: {
        "WIDTH": None,
        "HEIGHT": None  # WORD2_VEC

    }
}

# crop size
BEARD_CROP_WIDTH = 87
BEARD_CROP_HEIGHT = 71

HAIR_FRONT_CROP_WIDTH = 70
HAIR_FRONT_CROP_HEIGHT = 66

HAIR_LONG_CROP_WIDTH = 60
HAIR_LONG_CROP_HEIGHT = 124

HAIR_CURL_CROP_WIDTH = 100
HAIR_CURL_CROP_HEIGHT = 100

GLASSES_CROP_WIDTH = 100
GLASSES_CROP_HEIGHT = 54

# word2vec
WORD2VEC_CORPUS_FILENAME = 'corpus.txt'

# skin color table
SKIN_COLOR_TABLE_HEX = ["fef0e6",
                        "ffe8dd",
                        "fbe5d8",
                        "f6ceb6",
                        "e4ae8d",
                        "bc9278",
                        "a37558"
                        ]  # "ffeee3" last one is default
SKIN_COLOR_TABLE_RGB = [list(bytes.fromhex(color)) for color in SKIN_COLOR_TABLE_HEX]
SKIN_COLOR_TABLE_HSV = [list(colorsys.rgb_to_hsv(rgb[0] / 255, rgb[1] / 255, rgb[2] / 255)) for rgb in
                        SKIN_COLOR_TABLE_RGB]
# hear color
HAIR_COLOR_TABLE_HEX = ["7e7e7e",
                        "131f29",
                        "542f19",
                        "654a39",
                        "bc0000",
                        "3b0505",
                        "928d0f",
                        "0a2503",
                        "0063c0",
                        "191b37",
                        "d0d0d0",
                        "e4e4e4",
                        "3a3a3a"
                        ]  # 4c4842  last one is default
HAIR_COLOR_TABLE_RGB = [list(bytes.fromhex(color)) for color in HAIR_COLOR_TABLE_HEX]
HAIR_COLOR_TABLE_HSV = [list(colorsys.rgb_to_hsv(rgb[0] / 255, rgb[1] / 255, rgb[2] / 255)) for rgb in
                        HAIR_COLOR_TABLE_RGB]

# process
MAX_PROCESS = 3
# MAX_PROCESS = 1
MAX_WORKERS = 4

# full connected layer shape
FC_LAYER_SHAPE = {
    ToonstoryAI.BEARD.name: [128 * 9 * 11, 625],
    ToonstoryAI.CLAB.name: [128 * 16 * 16, 625],
    ToonstoryAI.DLIB.name: None,
    ToonstoryAI.GLASSES.name: [128 * 7 * 13, 625],
    ToonstoryAI.HAIR_CURL.name: [128 * 13 * 13, 625],
    ToonstoryAI.HAIR_FRONT.name: [128 * 9 * 9, 625],
    ToonstoryAI.HAIR_LONG.name: [128 * 16 * 8, 625],
    ToonstoryAI.TEXT_CNN.name: None,
    ToonstoryAI.WORD2VEC.name: None
}

DETECTOR_UPSCALING = 1
