import math
# import random


# import matplotlib.pyplot as plt
# import numpy as np
import PIL
# import PIL.ImageDraw
# import PIL.ImageFilter


def ViewFromDimsAndAngularWidth(dims, alpha_w):
    L = dims[0] / (math.tan(alpha_w / 2))
    return (alpha_w, 2 * math.atan(dims[1] / (2*L)))


class Moon:
    @classmethod
    def SizeFromView(cls, x, y, wv, view, **kwargs):
        L = wv / (math.tan(view[0] / 2))
        wm = 2 * L * math.tan(math.radians(0.25))
        return cls(x, y, int(math.ceil(wm/2)), view, **kwargs)

    def __init__(self, x, y, r, view, percent=35, angle=math.radians(20), mult_t=100, render_period=5, color=(200, 200, 160, 255)):
        self.__img = None
        self.__x0 = x
        self.__y0 = y
        self.__theta0 = math.atan(y)
        self.__x = x
        self.__y = y
        self.__angle = angle
        self.__view = view  # NOTE: THIS IS THE HORIZONTAL AND VERTICAL VIEW ANGLES....
        self.__mult_t = mult_t
        self.__t = 0
        self.__t_last_render = 0
        self.__render_period = render_period
        self.__color = color
        self.__percent = percent
        self.__rotation = math.radians(-10)
        self.__draw_moon(2 * r)

    @property
    def t(self):
        return self.__t

    @property
    def pos(self):
        return (self.__x, self.__y)

    @property
    def color(self):
        return self.__color

    @property
    def img(self):
        return self.__img

    @property
    def ill_kws(self):
        return {
            'x_src': self.__x,
            'h_src': self.__y,
            'color': self.__color
        }

    def calculate_mult_t(self, h_end, t_end):
        theta_end = math.atan(h_end)
        theta_start = math.atan(self.__y)
        dtheta = theta_end - theta_start
        t_real = dtheta / math.radians(4.1666666e-3)
        self.__mult_t = t_real / (t_end)
        print('Moon speed multiplier = {}'.format(self.__mult_t))
        return self.__mult_t

    def update(self, dt):
        self.__t += (self.__mult_t * dt)
        rerender = False
        theta = self.__theta0 + (self.__t * math.radians(4.1666666e-3))
        self.__y = math.tan(theta)
        dy = self.__y - self.__y0
        dx = dy * math.tan(self.__angle)
        self.__x = self.__x0 + dx
        if self.__t - self.__t_last_render > self.__render_period:
            self.__t_last_render += self.__render_period
            rerender = True
        return rerender

    def render(self, img, mask=None):
        w, h = self.__img.size
        wv, hv = img.size
        xc, yc = wv * self.__x, hv - (hv * self.__y)
        x = int(xc - (w/2) + 0.5)
        y = int(yc - (h/2) + 0.5)
        img.paste(self.__img, (x, y), mask=self.__img)
        if mask is not None:
            empty = PIL.Image.new('RGBA', img.size, color=(0, 0, 0, 0))
            img = PIL.Image.composite(img, empty, mask)
        return img

    def glow(self, dims, mask=None):
        w, h = self.__img.size
        wv, hv = dims
        length = 5 * w
        base = PIL.Image.new('RGBA', (length, length), color=(0, 0, 0, 0))
        base.paste(self.__img, (2*w, 2*h))
        xc, yc = wv * self.__x, hv - (hv * self.__y)
        x = int(xc - (length/2) + 0.5)
        y = int(yc - (length/2) + 0.5)
        if mask is not None:
            empty = PIL.Image.new('RGBA', (length, length), color=(0, 0, 0, 0))
            mask_crop = mask.crop((x, y, x+length, y+length))
            base = PIL.Image.composite(base, empty, mask_crop)
        N = int(2*w/3)
        base = base.filter(PIL.ImageFilter.MaxFilter(int(2*N + 1)))
        base = base.filter(PIL.ImageFilter.GaussianBlur(int(1.5 * N)))
        field = PIL.Image.new('RGBA', dims, color=(0, 0, 0, 0))
        field.paste(base, (x, y))
        return field

    def __draw_moon(self, d):
        unlit = (16, 16, 4, 255)
        lit = self.__color
        d = int(math.ceil(d))
        self.__img = PIL.Image.new('RGBA', (d, d), color=(0, 0, 0, 0))
        draw = PIL.ImageDraw.Draw(self.__img)

        mask = None
        fill_img = None
        draw.pieslice(((0, 0), (d-1, d-1)), 90, 270, fill=lit)
        draw.pieslice(((0, 0), (d-1, d-1)), 270, 90, fill=unlit)
        fill_img = PIL.Image.new('RGBA', (d, d), color=(0, 0, 0, 0))
        fill_draw = PIL.ImageDraw.Draw(fill_img)
        fill_draw.ellipse(((0, 0), (d-1, d-1)), fill=unlit if self.__percent <= 50 else lit)
        mask = PIL.Image.new('L', (d, d), 0)
        mask_draw = PIL.ImageDraw.Draw(mask)
        y_offset = int(d/25)
        percent_covered = self.__percent if self.__percent <= 50 else 100 - self.__percent
        w = max(2*y_offset, int(d * (1 - percent_covered / 50)))
        xc = d / 2
        x0 = int(xc - w/2)
        x1 = int(xc + w/2)
        mask_draw.ellipse(((x0, -y_offset), (x1, d+y_offset)), fill=255)
        mask = mask.filter(PIL.ImageFilter.GaussianBlur(y_offset))
        self.__img = PIL.Image.composite(fill_img, self.__img, mask)
        self.__img = self.__img.rotate(math.degrees(self.__rotation))


if __name__ == '__main__':
    import os
    import os.path

    import moviepy.editor as mpy

    import func_profiles
    import mountain_range
    import random_snowflake_plotting_test

    # fps = 24
    fps = 30
    t_total = 30
    dt = 1 / fps
    # dims = (1280, 720)
    dims = (1920, 1080)
    # dims = (640, 360)
    view = ViewFromDimsAndAngularWidth(dims, math.radians(20))

    moon = Moon.SizeFromView(0.25, 0.25, 1920, view, percent=75)
    moon.img.show()
    #
    # print('Generate Mountain Profile')
    # # random.seed(123456)
    #
    # background_img = random_snowflake_plotting_test.vertical_gradient((8, 8, 48, 255), (0, 0, 0, 255), dims, mode='RGBA')
    #
    # mountain_profile = func_profiles.TriangleProfile(x1=(0.6, 0.7), y1=1, y_mults=(0.15, 0.3))
    # mountain = mountain_range.MountainProfile(num_peaks=5, levels=7, scale_multiplier=0.27, profile_func=mountain_profile)
    # xs, hs = mountain.profile
    # print('Generate Mountain Mask')
    # mountain_mask = mountain_range.renderMountainMask(mountain, dims)
    # h_snow = mountain.snow_threshold(29, xs, hs)
    #
    # print('\t\tGenerate Moon')  # TODO: THIS SHOULD ACTUALLY BE REIMPLEMENTED WITH AN UPDATE ON THE MOON OBJECT
    # idx = int(len(xs) / 3)
    # x_moon_init, h_moon_init = xs[idx], 0.75*hs[idx]
    # moon = Moon.SizeFromView(x_moon_init, h_moon_init, dims[0], view, percent=40)
    # h_offset = moon.img.size[0] / (2 * dims[1])
    #
    # num_frames = fps * t_total
    # frame_directory = 'moon_frames'
    # output_directory = 'moon_outputs'
    # # idx = int(len(xs) / 3)
    # # x_moon_min, h_moon_min = xs[idx], 0.75*hs[idx]
    # # h_moon_max = 0.5
    # # idx = int(2 * len(xs) / 3)
    # # x_moon_max = xs[idx]
    # # h_moon_step = (h_moon_max - h_moon_min) / (num_frames-1)
    # # x_moon_step = (x_moon_max - x_moon_min) / (num_frames-1)
    #
    # print('Deleting previous frames')
    # for file in os.listdir(frame_directory):
    #     os.remove(os.path.join(frame_directory, file))
    #
    #
    # print('Generating {} Frames'.format(num_frames))
    # for idx in range(num_frames):
    #     print('\tGenerate Frame {}'.format(idx))
    #
    #     # print('Create Background')
    #     base = background_img.copy()  #PIL.Image.new('RGBA', dims, color=(0, 0, 0, 0))
    #     print('\t\tRender Moon')
    #     base = moon.render(base)
    #     glow = None
    #     # print('moon.pos = {}, h_offset = {}, mountain.h(x) = {}'.format(moon.pos, h_offset, mountain.h(moon.pos[0])))
    #     if (moon.pos[1]+h_offset) > mountain.h(moon.pos[0]):
    #         print('\t\tGenerate Glow')
    #         glow = moon.glow(dims, mask=mountain_mask)
    #     print('\t\tRender Mountains')
    #     ill_kws = moon.ill_kws
    #     ill_kws['strength'] = 0.3
    #     mid = mountain_range.renderMountain(mountain, dims, snow_threshold=h_snow, ill_kws=ill_kws)
    #     print('\t\tComposite Image')
    #     base = PIL.Image.alpha_composite(base, mid)
    #     if glow is not None:
    #         base = PIL.Image.alpha_composite(base, glow)
    #     print('\t\tUpdate Moon')
    #     update_landscape = moon.update(dt)
    #     filename = 'frame_{:05d}.png'.format(idx)
    #     print('\t\tSave frame -- {}'.format(filename))
    #     base.save(os.path.join(frame_directory, filename))
    #
    # print('Generate Animation...')
    # clip = mpy.ImageSequenceClip(frame_directory, fps=fps)
    # animation_path = os.path.join(output_directory, 'test_animation.mp4')
    # clip.write_videofile(animation_path, fps=fps)
    # print('Animation saved as {}'.format(animation_path))
