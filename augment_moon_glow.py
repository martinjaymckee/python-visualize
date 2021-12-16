import math
import os
import os.path
import random
import time

# import moviepy as mpy
# import moviepy.editor as mpe
import numpy as np
import PIL
import PIL.ImageDraw
import PIL.ImageEnhance
import PIL.ImageFilter
import PIL.ImageOps
import scipy
import scipy.interpolate

from colour_system import cs_srgb
import fractal_profile


cs = cs_srgb


class FrameProcessor:
    def __init__(self, fps, scale=1, frame_directory='augmented_moon_glow_frames'):
        self.__frame_num = 0
        self.__frame_directory = frame_directory
        self.__fps = fps
        self.__scale = scale
        self.__moon_fraction = 0.2
        self.__reference_area = 3000 / scale**2
        self.__add_moon_glow = True
        self.__bloom_transparency = 0.85
        self.__ring_transparency = 0.075
        self.__transparency_noise = 0.005
        self.__alpha_threshold = 50

        R = 650e-9
        O = 580e-9
        Y = 572e-9
        G = 550e-9
        C = 490e-9
        B = 465e-9
        V = 440e-9
        # dxs = [1/14] * 2 + [1/7] * 6
        # x = 0
        # xs = [x]
        # for dx in dxs:
        #     x += dx
        #     xs.append(x)
        # lams = [Y, O, R, V, B, G, Y, O, R]
        lams = [V, V, B, C, C, C, G, Y, Y, Y, O, O, O, R]
        xs = np.linspace(0, 1, len(lams))
        self.__color_spline = scipy.interpolate.UnivariateSpline(xs, lams)

    def save(self, output_directory, filename):
        # clip = mpe.ImageSequenceClip(self.__frame_directory, fps=self.__fps)
        # animation_path = os.path.join(output_directory, filename)
        # clip.write_videofile(animation_path, fps=self.__fps)
        # print('Animation saved as {}'.format(animation_path))
        pass

    def __call__(self, im, path, debug=False, t_start=None, frame_num=None):
        if t_start is None:
            t_start = time.perf_counter_ns()
        original_size = im.size
        if not self.__scale == 1:
            im = im.resize((int(im.size[0]/self.__scale), int(im.size[1]/self.__scale)))
        frame_num = self.__frame_num if frame_num is None else frame_num
        print('\tFrame {} ({})'.format(frame_num, path))

        color_contrast_enhancer = PIL.ImageEnhance.Contrast(im)
        im = color_contrast_enhancer.enhance(1.5)
        row_array = np.array(range(im.size[0]))
        column_array = np.array(range(im.size[1]))
        alpha = PIL.ImageOps.grayscale(im)
        alpha_contrast_enhancer = PIL.ImageEnhance.Contrast(alpha)
        alpha = alpha_contrast_enhancer.enhance(2)
        alpha_array = np.array(alpha)
        alpha_array[alpha_array < 128] = 0
        alpha = PIL.Image.fromarray(alpha_array)
        im.putalpha(alpha)
        im_array = np.array(im)
        pos, area, diameter, moon_color = self.__estimate_moon_params(im_array, row_array, column_array)
        if debug:
            print('Mean Color = {}'.format(moon_color))
            print('Center Position = {}'.format(pos))
            print('Area = {}'.format(area))
        # print(im_array.shape)
        composite = PIL.Image.new('RGBA', im.size, color=(0, 0, 0, 0))
        ring_alpha = int(255 * (min(1, max(0, self.__ring_transparency + random.gauss(0, self.__transparency_noise)))) * (area / self.__reference_area))
        if not area == 0:
            if self.__add_moon_glow:
                glow = PIL.Image.new('RGBA', im.size, color=(255, 255, 255, 0))
                im_array[im_array[:, :, 3] < self.__alpha_threshold] = (255, 255, 255, 0)
                im = PIL.Image.fromarray(im_array)
                im = im.filter(PIL.ImageFilter.GaussianBlur(int(10 / self.__scale)))
                composite = PIL.Image.blend(composite, im, self.__bloom_transparency)
                draw = PIL.ImageDraw.Draw(glow)
                r = 1.6 * diameter
                x0, y0 = (int(pos[0] - r), int(pos[1] - r))
                x1, y1 = (int(pos[0] + r), int(pos[1] + r))
                glow_transparency = min(255, int(1.25*ring_alpha))
                glow_color = (255, 255, 255, glow_transparency)
                draw.ellipse([(x0, y0), (x1, y1)], fill=glow_color)
                glow = glow.filter(PIL.ImageFilter.GaussianBlur(int(50 / self.__scale)))
                composite = PIL.Image.alpha_composite(composite, glow)

            levels = 4
            scale_multiplier = 0.55
            var_outer = 0.1 * 3.35 * diameter
            var_inner = 0.1 * 1.65 * diameter
            r_outer = 3.35 * diameter - math.sqrt(sum([(scale_multiplier**(1+idx) * var_outer)**2 for idx in range(levels)]))
            r_inner = 1.65 * diameter - math.sqrt(sum([(scale_multiplier**(1+idx) * var_inner)**2 for idx in range(levels)]))
            kwargs = {
                'levels': levels,
                'num_peaks': 30,
                'frequency_multiplier': 2.35,
                'scale_multiplier': scale_multiplier
            }
            outer_profile = fractal_profile.FractalProfile.Circular(ref_height=var_outer, offset_height=r_outer, **kwargs)
            kwargs['num_peaks'] = 15
            inner_profile = fractal_profile.FractalProfile.Circular(ref_height=var_inner, offset_height=r_inner, **kwargs)
            outer_radius_extents = outer_profile.extents()

            r_max = 1.01 * outer_radius_extents[1]

            ring = PIL.Image.new('RGBA', im.size, color=(255, 255, 255, 0))
            if debug:
                draw = PIL.ImageDraw.Draw(ring, mode='RGBA')
                x0, y0 = (0, int(pos[1]))
                x1, y1 = (im.size[0], int(pos[1]))
                draw.line([(x0, y0), (x1, y1)], fill = (255, 0, 0, 255))
                x0, y0 = (int(pos[0]), 0)
                x1, y1 = (int(pos[0]), im.size[1])
                draw.line([(x0, y0), (x1, y1)], fill = (255, 0, 0, 255))
                r = 3.5 * diameter
                fill_color = self.__calc_ring_color(0.5, 0, 1, ring_alpha)
                x0, y0 = (int(pos[0] - r), int(pos[1] - r))
                x1, y1 = (int(pos[0] + r), int(pos[1] + r))
                draw.ellipse([(x0, y0), (x1, y1)], fill = fill_color)
                r = 1.75 * diameter
                x0, y0 = (int(pos[0] - r), int(pos[1] - r))
                x1, y1 = (int(pos[0] + r), int(pos[1] + r))
                draw.ellipse([(x0, y0), (x1, y1)], fill = (255, 255, 255, 0))

            xc, yc = pos
            x0, y0 = self.__constrain_pos_to_canvas((xc - r_max, yc - r_max), im)
            x1, y1 = self.__constrain_pos_to_canvas((xc + r_max, yc + r_max), im)

            pixdata = ring.load()
            for x in range(x0, x1):
                # print('{} of {}'.format(x - x0, x1-x0))
                for y in range(y0, y1):
                    theta = math.atan2(x-xc, y-yc)
                    r = math.sqrt(((x - xc)**2) + ((y - yc)**2))
                    r_o = outer_profile.params(theta)[0]
                    r_i = inner_profile.params(theta)[0]
                    if r_i <= r <= r_o:
                        color = self.__calc_ring_color(r, r_o, r_i, ring_alpha)
                        pixdata[x, y] = color
            ring = ring.filter(PIL.ImageFilter.GaussianBlur(5))
            composite = PIL.Image.alpha_composite(composite, ring)
        # composite.show()
        print('\t\tin {} s'.format((time.perf_counter_ns() - t_start) / 1e9))
        im_filename = 'frame_{:05d}.png'.format(frame_num)
        if not self.__scale == 1:
            composite = composite.resize(original_size)
            composite = composite.filter(PIL.ImageFilter.GaussianBlur(int(self.__scale)))
        composite.save(os.path.join(self.__frame_directory, im_filename))
        print('\t\t stored as {}'.format(im_filename))
        self.__frame_num += 1

        return

    def __estimate_moon_params(self, im_array, row_array, column_array):
        reds, greens, blues = im_array[:, :, 0], im_array[:, :, 1], im_array[:, :, 2]
        alphas = im_array[:, :, 3]
        alphas_sum = np.sum(alphas)
        if not alphas_sum == 0:
            mean_reds = np.average(reds, weights=alphas)
            mean_greens = np.average(greens, weights=alphas)
            mean_blues = np.average(blues, weights=alphas)
            mean_color = (mean_reds, mean_greens, mean_blues)
            rows_center = np.average(row_array, weights=np.sum(alphas, axis=0))
            row_alpha_sums = np.sum(alphas, axis=0)
            alt_rows_center = np.mean(row_array *  row_alpha_sums / np.sum(row_alpha_sums))
            columns_center = np.average(column_array, weights=np.sum(alphas, axis=1))
            opaque = np.zeros(alphas.shape)
            opaque[alphas > self.__alpha_threshold] = 1
            area = np.sum(opaque)
            total_area = area / self.__moon_fraction
            diameter = 2 * math.sqrt(total_area / math.pi)
            pos = (rows_center + 0.35 * diameter, columns_center)  # TODO: THIS SHOULD BE BASED ON THE ACTUAL MOON FRACTION....
            return pos, area, diameter, mean_color
        return None, 0, None, None

    def __calc_ring_color(self, r, r_o, r_i, ring_alpha):
        def r_to_lam():
            p = (r - r_i) / (r_o - r_i)
            if p > (1/4):  # Outer Ring
                return self.__color_spline((4/3) * (p-1/4))
            return self.__color_spline((2*p)+(1/2))  # Inner Ring
        lam = r_to_lam()
        rgb = cs.lam_to_rgb(lam)
        color = (int(255 * rgb[0]), int(255 * rgb[1]), int(255 * rgb[2]), ring_alpha)
        return color

    def __constrain_pos_to_canvas(self, pos, im):
        x = max(0, min(pos[0], im.size[0]))
        y = max(0, min(pos[1], im.size[1]))
        return (int(x), int(y))


def getFrameNum(file):
    name, ext = os.path.splitext(file)
    _, _, frame_num = name.partition('_')
    return int(frame_num)


if __name__ == '__main__':
    print('Frame Number = {}'.format(getFrameNum('frame_01000.png')))

    fps = 60
    scale = 5
    frame_range = (13500, 13501)
    output_directory = 'snowy_landscape_outputs'
    input_directory = 'moon_reference_frames'
    input_filename = '4k_animation_moon_reference_3_45s_2.mp4'
    output_filename = 'test_augmented_moon_output.mp4'
    print('Load Moon Reference Video')

    processor = FrameProcessor(fps, scale=scale)
    print('Generate Augmented Frames')
    for file in os.listdir(input_directory):
        frame_num = getFrameNum(file)
        if (frame_range is None) or (frame_range[0] <= frame_num <= frame_range[1]):
            path = os.path.join(input_directory, file)
            t_start = time.perf_counter_ns()
            im = PIL.Image.open(path)
            processor(im, path, debug=False, t_start=t_start, frame_num=frame_num)

    #processor.save(output_directory, output_filename)

    # print('Generate Main Animation...')
    # clip = mpy.ImageSequenceClip(frame_directory, fps=fps)
    # animation_path = os.path.join(output_directory, 'test_animation.mp4')
    # clip.write_videofile(animation_path, fps=fps)
    # print('Animation saved as {}'.format(animation_path))
