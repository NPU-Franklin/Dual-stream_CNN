import argparse
import os
import sys
import xml.etree.ElementTree as ET

import cv2
import numpy as np
from tqdm import tqdm


class Preprocessor:
    def __init__(self, input, output):
        self.input = input
        self.output = output
        self.filenames = os.listdir(self.input)
        try:
            self.filenames.remove("Masks")
        except ValueError:
            pass
        try:
            self.filenames.remove("Edges")
        except ValueError:
            pass

    def xml2mask_and_edge(self, filename):
        mask = np.zeros([1000, 1000], dtype=np.uint8)
        edge = [[255 for i in range(0, 1000)] for i in range(0, 1000)]
        edge = np.asarray(edge, dtype=np.uint8)

        xml = str(self.input + "/" + filename + ".xml")
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
        if not os.path.exists(self.output + "/Masks"):
            os.mkdir(self.output + "/Masks")
        if not os.path.exists(self.output + "/Edges"):
            os.mkdir(self.output + "/Edges")
        cv2.imwrite(self.output + "/Masks/" + filename + ".png", mask)
        cv2.imwrite(self.output + "/Edges/" + filename + ".png", edge)


def parse_args(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", type=str, help="directory to annotations", default="./")
    parser.add_argument("--output", type=str, help="output directory", default="./")

    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    input = args.input
    output = args.output

    preprocessor = Preprocessor(input, output)
    split_filenames = []
    for filename in preprocessor.filenames:
        split_filename, _ = os.path.splitext(filename)
        split_filenames.append(split_filename)

    for filename in tqdm(split_filenames):
        preprocessor.xml2mask_and_edge(filename)
    print("Done!")
