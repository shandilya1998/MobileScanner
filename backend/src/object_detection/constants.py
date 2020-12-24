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
LABELS = (
    'TextBlock',
    'Drawing',
    'Diagram',
    'List',
    'Formula',
    'Table',
)
IMAGE_H, IMAGE_W = 1024, 768
NUM_C = 3
CLASS            = len(LABELS)
BATCH_SIZE       = 5
EPOCHS           = 200


# Train and validation directory

train_image_folder = '../../data/object_detection/images/train/'
train_annot_folder = '../..//data/object_detection/annotations/train/'
val_image_folder = '../../data/object_detection/images/val/'
val_annot_folder = '../../data/object_detection/annotations/val/'

GRID_H,  GRID_W  = 32, 24 # GRID size = IMAGE size / 32
BOX              = 5 #12 # 5
SCORE_THRESHOLD  = 0.5
IOU_THRESHOLD    = 0.45
ANCHORS          = [0.04907460632666401, 0.02944813150316293, 0.48454704726922426, 0.4774729075500347, -0.2334564834010921, -0.2547636770925863, 0.18407737068116317, 0.17192014203873957, 0.6749850295885473, 0.6654344207918693, -0.09109767083442549, -0.10144362002906732, 0.32452719433062316, 0.32168271146404637, -0.39470880681820975, -0.42300047538256125, 0.1067620735223449, -0.3285999068957185, 0.5313478059530009, 0.08406022451456363]

LAMBDA_NOOBJECT  = 1
LAMBDA_OBJECT    = 5
LAMBDA_CLASS     = 1
LAMBDA_COORD     = 1

max_annot        = 0
EPSILON          = 1e-7
