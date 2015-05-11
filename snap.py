import math
import cmath

from PIL import Image
import numpy


def normalize(a, outliers=0.05):
    assert outliers > 0
    sorted = numpy.sort(a.flatten())
    min_value = sorted[int(len(sorted) * outliers)]
    max_value = sorted[int(len(sorted) * (1 - outliers))]
    if max_value > min_value:
        scale = 1.0 / (max_value - min_value)
    else:
        scale = 0
    return numpy.clip((a - min_value) * scale, 0, 1)


def project_tilt(a, slope):
    h, w = a.shape

    offsets = numpy.floor(-numpy.arange(h) * slope + 0.5).astype(int)
    offsets -= offsets.min()

    result = numpy.zeros(w + offsets.max())

    for i in range(h):
        result[offsets[i]: offsets[i] + w] += a[i]

    return result


def best_tilt(a, min_slope, max_slope, steps):
    slopes = numpy.arange(steps) * (max_slope - min_slope) / (steps - 1) + min_slope
    return max(slopes, key=lambda slope: (project_tilt(a, slope) ** 2).sum())


def remove_tilt(img, slope):
    w, h = img.size
    new_w = int(w + abs(slope) * h)
    return img.transform(
        (new_w, h), Image.AFFINE,
        (1, slope, -max(h * slope, 0), 0, 1, 0),
        Image.BILINEAR)


def best_period(a, min_period, max_period, steps):
    assert a.ndim == 1, a.ndim
    a = a * a

    periods = numpy.exp(
        numpy.arange(steps) *
        (math.log(max_period) - math.log(min_period)) / steps) * min_period

    best_score = float('-inf')
    best_peak = 0
    best_period = 0
    for period in periods:
        s = numpy.dot(
            a, numpy.exp(numpy.arange(len(a)) / period * 2 * math.pi * 1j))
        score = abs(s)
        if score > best_score:
            best_score = score
            best_period = period
            best_peak = (cmath.phase(s) / (2 * math.pi) * period) % period

    return best_period, best_peak


def align_period(img, period, peak, new_period):
    w, h = img.size
    new_w = int(int(w / period + 2) * new_period)
    return img.transform(
        (new_w, h), Image.AFFINE,
        (period / new_period, 0, peak - period, 0, 1, 0),
        Image.BILINEAR)


def default_edges_img(img):
    e = edges_(img)
    e = normalize(e)
    e = Image.fromarray((255 * e).astype(numpy.uint8))
    return e


def snap_to_vertical_grid(img, block_size=16):
    edges_img = default_edges_img(img)
    edges = numpy.array(edges_img).astype(float)
    assert edges.ndim == 2, edges.ndim

    # TODO: propagate parameters
    tilt = best_tilt(edges, -0.5, 0.5, 400)

    img = remove_tilt(img, tilt)
    edges_img = remove_tilt(edges_img, tilt)

    edges = numpy.array(edges_img).astype(float)
    assert edges.ndim == 2, edges.ndim

    a = edges.sum(axis=0)
    a = normalize(a)
    period, peak = best_period(a, 3, img.size[0] / 6, 300)
    img = align_period(img, period, peak, block_size)

    return img


def snap_to_grid(img, block_size=16):

    def transpose(img):
        return img.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.ROTATE_90)
        # return img.transform(
        #     (img.size[1], img.size[0]), Image.AFFINE,
        #     (0, 1, 0, 1, 0, 0),
        #     Image.NEAREST)

    for _ in range(2):
        img = snap_to_vertical_grid(img, block_size)
        img = transpose(img)

    return img


def edges_(img):
    # Workaround for apparent bug in Pillow. Grayscale images with alpha channel
    # do not convert to arrays directly for some reason.
    if img.mode == 'LA':
        img = img.convert('L')

    a = numpy.array(img)

    # TODO: make it same size as original image.
    d = a[:,1:] - a[:,:-1]

    if d.ndim == 2:
        return d * d
    elif d.ndim == 3:
        # get rid of alpha
        if d.shape[2] in [2, 4]:
            d = d[...,:-1] * a[:,1:,[-1]]
        assert d.shape[2] in [1, 3]
        return (d * d).sum(axis=2)
    else:
        assert False, d.ndim
