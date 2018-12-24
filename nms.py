'''
Description: NMS

Author: YaoLing
Github: https://github.com/YaoLing13
'''

import numpy as np

def iou_calc(boxes1, boxes2, type=3):
    """
    calculate iou value between box1 and box2.

    :param boxes1: [xmin, ymin, xmax, ymax, score, class]
    :param boxes2: [xmin, ymin, xmax, ymax, score, class]
    :param type: iou calculate type, 1-union, 2-max_area, 3-min_area
    :return: iou value
    """
    boxes1_area = (boxes1[2] - boxes1[0]) * (boxes1[3] - boxes1[1])
    boxes2_area = (boxes2[2] - boxes2[0]) * (boxes2[3] - boxes2[1])

    left_up = np.maximum(boxes1[:2], boxes2[:2])
    right_down = np.minimum(boxes1[2:-2], boxes2[2:-2])

    inter_section = np.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[0] * inter_section[1]
    if type == 1:
        union_area = boxes1_area + boxes2_area - inter_area
    elif type == 2:
        union_area = max(boxes1_area, boxes2_area)
    elif type == 3:
        union_area = min(boxes1_area, boxes2_area)
    else:
        union_area = 99999

    IOU = 1.0 * inter_area / union_area
    return IOU


def nms(boxes, iou_threshold=0.9):
    """
    nms function

    :param boxes: list of box
    :param iou_threshold:
    :return:
    """
    return_box = []
    if len(boxes) > 0:
        boxes_dict = {}
        for box in boxes:
            if box[5] in boxes_dict:
                boxes_dict[box[5]].append(box)
            else:
                boxes_dict[box[5]] = [box]

        for boxs in boxes_dict.values():
            if len(boxs) == 1:
                return_box.append(boxs[0])
            else:
                while(len(boxs)):
                    best_box = boxs.pop(0)
                    return_box.append(best_box)
                    j = 0
                    for i in range(len(boxs)):
                        i -= j
                        # print(best_box)
                        # print(boxs[i])
                        if iou_calc(best_box, boxs[i]) > iou_threshold:
                            boxs.pop(i)
                            j += 1
    return return_box