import argparse
import os
import sys
import xml.etree.ElementTree as ET

import cv2
import numpy as np
from tqdm import tqdm


def xml2mask_and_edge(dir, filename):
    mask = np.zeros([1000, 1000], dtype=np.uint8)
    edge = [[255 for i in range(0, 1000)] for i in range(0, 1000)]
    edge = np.asarray(edge, dtype=np.uint8)

    xml = str(dir + "/" + filename + ".xml")
    tree = ET.parse(xml)
    root = tree.getroot()
    regions = root.findall("Annotation/Regions/Region")
    for region in regions:
        points = []
        for point in region.findall("Vertices/Vertex"):
            x = float(point.attrib["X"])
            y = float(point.attrib["Y"])
            points.append([x, y])

        points = np.asarray([points], dtype=np.int32)
        cv2.fillPoly(img=mask, pts=points, color=255)
        cv2.polylines(img=edge, pts=points, color=0, thickness=2, isClosed=True)
    if not os.path.exists(dir + "/Masks"):
        os.mkdir(dir + "/Masks")
    if not os.path.exists(dir + "/Edges"):
        os.mkdir(dir + "/Edges")
    cv2.imwrite(dir + "/Masks/" + filename + ".png", mask)
    cv2.imwrite(dir + "/Edges/" + filename + ".png", edge)


def parse_args(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument("--dir", type=str, help="directory to annotations", default="./")

    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    dir = args.dir
    filenames = os.listdir(dir)
    try:
        filenames.remove('Masks')
    except ValueError:
        pass
    try:
        filenames.remove('Edges')
    except ValueError:
        pass

    split_filenames = []
    for filename in filenames:
        split_filename, _ = os.path.splitext(filename)
        split_filenames.append(split_filename)


    def feed_file(files):
        for file in files:
            yield file


    feeder = feed_file(split_filenames)
    for i in tqdm(range(0, len(split_filenames))):
        xml2mask_and_edge(dir, next(feeder))
