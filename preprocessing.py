import argparse
import os
import sys
import xml.etree.ElementTree as ET

import cv2
import numpy as np
from tqdm import tqdm


class Preprocessor:
    def __init__(self, input, output, type):
        self.input = input
        self.output = output
        self.type = type
        self.filenames = os.listdir(self.input)
        try:
            self.filenames.remove("Masks")
        except ValueError:
            pass
        try:
            self.filenames.remove("Edges")
        except ValueError:
            pass
        if not os.path.exists(self.output):
            os.mkdir(self.output)

    def rotate(self, filename):
        img = cv2.imread(self.output + "/Tissue Images/" + filename + self.type)
        mask = cv2.imread(self.output + "/Masks/" + filename + ".png")
        edge = cv2.imread(self.output + "/Edges/" + filename + ".png")

        new_filenames = []
        for image in [img, mask, edge]:
            for angle in range(30, 270, 30):
                rotate_matrix = np.array([[np.cos(angle), -np.sin(angle), 0],
                                          [np.sin(angle), np.cos(angle), 0],
                                          [0, 0, 1]])
                center = np.array((0.5 * img.shape[0], 0.5 * img.shape[1]))
                pos_basic_matrix = np.array([[1, 0, center[0]],
                                             [0, 1, center[1]],
                                             [0, 0, 1]])
                neg_basic_matrix = np.array([[1, 0, -center[0]],
                                             [0, 1, -center[1]],
                                             [0, 0, 1]])
                rotate_matrix = np.linalg.multi_dot([pos_basic_matrix, rotate_matrix, neg_basic_matrix])
                out_img = cv2.warpAffine(image, rotate_matrix[:2, :], dsize=(img.shape[1], img.shape[0]),
                                         flags=cv2.INTER_LINEAR,
                                         borderMode=cv2.BORDER_REFLECT, borderValue=0)
                if image is img:
                    new_filename = str(self.output + "/Tissue Images/" + filename + "_{}".format(angle) + ".png")
                elif image is mask:
                    new_filename = str(self.output + "/Masks/" + filename + "_{}".format(angle) + ".png")
                else:
                    new_filename = str(self.output + "/Edges/" + filename + "_{}".format(angle) + ".png")
                cv2.imwrite(new_filename, out_img)
                new_filenames.append(filename + "_{}".format(angle))
        return new_filenames

    def xml2mask_and_edge(self, filename):
        mask = np.zeros([1000, 1000], dtype=np.uint8)
        edge = np.zeros([1000, 1000], dtype=np.uint8)

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
            cv2.polylines(img=edge, pts=points, color=255, thickness=4, isClosed=True)

        if not os.path.exists(self.output + "/Masks"):
            os.mkdir(self.output + "/Masks")
        if not os.path.exists(self.output + "/Edges"):
            os.mkdir(self.output + "/Edges")
        cv2.imwrite(self.output + "/Masks/" + filename + ".png", mask)
        cv2.imwrite(self.output + "/Edges/" + filename + ".png", edge)

    def patch(self, filename, patch_size):
        img = cv2.imread(self.output + "/Tissue Images/" + filename + self.type)
        img = np.pad(img, ((12, 12), (12, 12), (0, 0)), "symmetric")

        mask = cv2.imread(self.output + "/Masks/" + filename + ".png")
        mask = np.pad(mask, ((12, 12), (12, 12), (0, 0)), "symmetric")

        edge = cv2.imread(self.output + "/Edges/" + filename + ".png")
        edge = np.pad(edge, ((12, 12), (12, 12), (0, 0)), "symmetric")

        num_rows = int((img.shape[0] - 256) / 128) + 1
        num_cols = int(img.shape[1] // 256)

        if not os.path.exists(self.output + "/Patch"):
            os.mkdir(self.output + "/Patch")
        if not os.path.exists(self.output + "/Patch/Imgs"):
            os.mkdir(self.output + "/Patch/Imgs")
        if not os.path.exists(self.output + "/Patch/Masks"):
            os.mkdir(self.output + "/Patch/Masks")
        if not os.path.exists(self.output + "/Patch/Edges"):
            os.mkdir(self.output + "/Patch/Edges")

        for r in range(num_rows):
            for c in range(num_cols):
                patch_img = img[r * 128: r * 128 + patch_size, c * 256: c * 256 + patch_size, :]
                patch_mask = mask[r * 128: r * 128 + patch_size, c * 256: c * 256 + patch_size, :]
                patch_edge = edge[r * 128: r * 128 + patch_size, c * 256: c * 256 + patch_size, :]

                cv2.imwrite(self.output + "/Patch/Imgs/" + filename + str(r) + "_" + str(c) + ".png", patch_img)
                cv2.imwrite(self.output + "/Patch/Masks/" + filename + str(r) + "_" + str(c) + ".png", patch_mask)
                cv2.imwrite(self.output + "/Patch/Edges/" + filename + str(r) + "_" + str(c) + ".png", patch_edge)


def parse_args(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--input", type=str, dest="input", help="directory to annotations", default="./")
    parser.add_argument("-o", "--output", type=str, dest="output", help="output directory", default="./")
    parser.add_argument("-t", "--type", type=str, dest="type", help="type of images", default=".png")

    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    input = str(args.input)
    output = str(args.output)
    type = str(args.type)

    preprocessor = Preprocessor(input, output, type)
    split_filenames = []
    for filename in preprocessor.filenames:
        split_filename, _ = os.path.splitext(filename)
        split_filenames.append(split_filename)

    new_filenames = []
    for filename in tqdm(split_filenames):
        preprocessor.xml2mask_and_edge(filename)
        new = preprocessor.rotate(filename)
        new_filenames += new

    for filename in tqdm(set(split_filenames + new_filenames)):
        preprocessor.patch(filename, 256)
    print("Done!")
