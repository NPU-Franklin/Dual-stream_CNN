import argparse
import logging
import os

import cv2
import matlab
import matlab.engine
import numpy as np
from tqdm import tqdm

from eval import metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate predict results using Hausdorff matrix",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-i', '--input', dest='input', type=str, default='', help='input dir')
    parser.add_argument('-m', '--matrix', dest='matrix', type=str, default='Hausdorff', help='evaluation matrix, '
                                                                                             'include: Hausdorff, '
                                                                                             'AJI, brp_AJI')

    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')

    args = parse_args()
    input = str(args.input)
    if input == '':
        raise ValueError('Input folder unspecified')
    matrix = str(args.matrix)
    logging.info("Loading images from {}".format(input))

    eng = matlab.engine.start_matlab()
    eng.cd("./eval")
    logging.info("Matlab engine start")

    imgs = os.listdir(input + "/true")
    logging.info("Input {} images".format(len(imgs)))
    logging.info("Evaluation matrix: {}".format(matrix))

    with tqdm(total=len(imgs), desc="Eval", unit='img') as pbar:
        scores = []
        for img in imgs:
            pred = cv2.imread(input + "/pred/" + img)
            true = cv2.imread(input + "/true/" + img)
            if matrix == "Hausdorff":
                scores.append(eng.ObjectHausdorff(matlab.single(pred.tolist()), matlab.single(true.tolist())))
            elif matrix == "AJI":
                scores.append(
                    eng.Aggregated_Jaccard_Index_v1_0(matlab.single(true.tolist()), matlab.single(pred.tolist())))
            elif matrix == "brp_AJI":
                scores.append(metrics.get_fast_aji(metrics.remap_label(true), metrics.remap_label(pred)))
            else:
                raise ValueError("Unsupported matrix {}".format(matrix))
            pbar.update()
    score = np.array(scores).mean()
    min = min(scores)
    max = max(scores)
    logging.info("{} Score: {}".format(matrix, score))
    logging.info("Range: {} ~ {}".format(min, max))

    eng.exit()
