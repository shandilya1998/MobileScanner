LABELS_ALL = (
    'TextBlock',
    'TextLine',
    'Word',
    'Correction',
    'Drawing',
    'Diagram',
    'Structure',
    'List',
    'Formula',
    'Table',
    'Symbol',
    'Arrow'
)
LABELS_SIMPLIFIED = (
    'TextBlock',
    'Drawing',
    'Diagram',
    'List',
    'Formula',
    'Table',
)

LABELS = (
    'TextLine',
    'Drawing',
    'Table',
    'Formula',
    'Structure',
)

IMAGE_H = 1024
IMAGE_W =  768
NUM_C = 3
CLASS            = len(LABELS)
BATCH_SIZE       = 5
EPOCHS           = 200

# Train and validation directory

train_image_folder = '../../data/object_detection/images/train/'
train_annot_folder = '../../data/object_detection/annotations/train/'
val_image_folder = '../../data/object_detection/images/val/'
val_annot_folder = '../../data/object_detection/annotations/val/'
i2c = pickle.load(open('../../data/object_detection/index2character.pickle', 'rb'))
c2i = pickle.load(open('../../data/object_detection/character2index.pickle', 'rb'))


GRID_H = 32
GRID_W = 24

scale_w = IMAGE_W / GRID_W
scale_h = IMAGE_H / GRID_H
BOX              = 5 #12 # 5
SCORE_THRESHOLD  = 0.5
IOU_THRESHOLD    = 0.45
ANCHORS          = [0.04907460632666401, 0.02944813150316293, 0.48454704726922426, 0.4774729075500347, -0.2334564834010921, -0.2547636770925863, 0.18407737068116317, 0.17192014203873957, 0.6749850295885473, 0.6654344207918693]

LAMBDA_NOOBJECT  = 1
LAMBDA_OBJECT    = 5
LAMBDA_CLASS     = 1
LAMBDA_COORD     = 1

max_annot        = 0
EPSILON          = 1e-7
%load_ext tensorboard
import warnings
warnings.filterwarnings("ignore")
!pip install imgaug==0.4.0

MAX_TEXT_LENGTH = 20
