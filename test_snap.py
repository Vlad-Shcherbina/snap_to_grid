import glob
import os

from PIL import Image, ImageDraw
import numpy

import snap


def draw_grid(img, block_size):
    w, h = img.size
    draw = ImageDraw.Draw(img)
    for i in range(0, w, block_size):
        draw.line((i, 0, i, h), fill=(255, 0, 0))
    for i in range(0, h, block_size):
        draw.line((0, i, w, i), fill=(255, 0, 0))


def main():
    block_size = 16
    for test_file in sorted(glob.glob('testdata/*.png')):
        print(test_file)
        result_file = os.path.join('testresults', os.path.basename(test_file))

        img = Image.open(test_file)

        img = snap.snap_to_grid(img, block_size=block_size)

        img = img.convert('RGB')

        a = numpy.array(img)
        for i in range(0, a.shape[0], block_size):
            for j in range(0, a.shape[1], block_size):
                block = a[i: i + block_size, j: j + block_size].astype(float)
                a[i: i + block_size, j: j + block_size] = \
                    block.sum(axis=0).sum(axis=0) / block_size ** 2
        blocked_img = Image.fromarray(a)

        draw_grid(img, block_size=block_size)

        result = Image.new(size=(1 * img.size[0], img.size[1]), mode='RGB')
        result.paste(img, (0, 0))
        result.paste(blocked_img, (img.size[0], 0))
        result.save(result_file)


if __name__ == '__main__':
    main()
