import argparse
import logging
import os

import cv2
import matlab
import matlab.engine
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate predict results using Hausdorff matrix",
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('-i', '--input', dest='input', type=str, default='', help='input dir')

    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')

    args = parse_args()
    input = str(args.input)
    if input == '':
        raise ValueError('Input folder unspecified')
    logging.info("Loading images from {}".format(input))

    eng = matlab.engine.start_matlab()
    eng.cd("./eval")
    logging.info("Matlab engine start")

    imgs = os.listdir(input + "/true")
    logging.info("Input {} images".format(len(imgs)))

    with tqdm(total=len(imgs), desc="Eval", unit='img') as pbar:
        total = 0
        for img in imgs:
            pred = cv2.imread(input + "/pred/" + img).tolist()
            true = cv2.imread(input + "/true/" + img).tolist()
            total += eng.ObjectHausdorff(matlab.single(pred), matlab.single(true))
            pbar.update()
        score = total / len(imgs)
        print("Hausdorff Score: {}".format(score))

    eng.exit()
