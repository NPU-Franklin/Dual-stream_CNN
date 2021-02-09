import argparse
import logging
import os

import cv2
import matlab
import matlab.engine
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate predict results using Hausdorff matrix",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-i', '--input', dest='input', type=str, default='', help='input dir')
    parser.add_argument('-m', '--matrix', dest='matrix', type=str, default='Hausdorff', help='evaluation matrix')

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
        total = 0
        for img in imgs:
            pred = cv2.imread(input + "/pred/" + img).tolist()
            true = cv2.imread(input + "/true/" + img).tolist()
            if matrix == "Hausdorff":
                total += eng.ObjectHausdorff(matlab.single(pred), matlab.single(true))
            elif matrix == "AJI":
                total += eng.Aggregated_Jaccard_Index_v1_0(matlab.single(true), matlab.single(pred))
            else:
                raise ValueError("Unsupported matrix {}".format(matrix))
            pbar.update()
        score = total / len(imgs)
        logging.info("{} Score: {}".format(matrix, score))

    eng.exit()
