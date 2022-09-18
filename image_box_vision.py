import matplotlib.pyplot as plt
from utils.anchor_utils import get_anchors
from utils.dataloader import FRCNNDatasets

if __name__ == '__main__':
    TRAIN_ANNOTATION_PATH = 'train.txt'
    INPUT_SHAPE = (600, 600)

    anchors = get_anchors(input_shape=(600, 600), sizes=[128, 256, 512])
    anchors = anchors * 600
    with open(TRAIN_ANNOTATION_PATH, 'r', encoding='utf-8') as f:
        train_lines = f.readlines()

    db = FRCNNDatasets(train_lines, INPUT_SHAPE, anchors, 1, 21, train=False)
    image_data, boxes = db.process_data(train_lines[0], INPUT_SHAPE, random=False)

    fig = plt.figure()
    ax = plt.subplot(111)
    plt.imshow(image_data / 255)
    for box in boxes:
        ax.add_patch(plt.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], color='r', fill=False))

    for i in range(len(anchors)):
        anchor = anchors[i]
        if i>37*18*9 and i<37*18*9 + 9:
            ax.add_patch(plt.Rectangle((anchor[0], anchor[1]), anchor[2] - anchor[0], anchor[3] - anchor[1], color='b', fill=False))
    plt.show()


