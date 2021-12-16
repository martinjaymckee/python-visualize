import math
import random


import matplotlib.pyplot as plt
import numpy as np
import PIL
import PIL.ImageDraw
import PIL.ImageFilter

import func_profiles

def blend(c1, c2, a=0.5):
    c1 = np.array(c1)
    c2 = np.array(c2)
    cr = a * c1 + (1-a) * c2
    return tuple([int(v) for v in cr])


class MountainProfile:
    def __init__(self, num_peaks=7, ref_height=0.25, scale_multiplier=0.3, frequency_multiplier=2.5, levels=7, profile_func=None):
        self.__num_peaks = num_peaks
        self.__ref_height = ref_height
        self.__scale_multiplier = scale_multiplier
        self.__frequency_multiplier = frequency_multiplier
        self.__levels = levels
        self.__profile_func = profile_func
        self.__profiles = []
        self.__generate_profiles()
        self.__xs = None
        self.__hs = None
        self.__ms = None
        self.__bs = None

    @property
    def profile(self):
        if self.__xs is None:
            return self.__interpolate_profiles(self.__profiles)
        return self.__xs, self.__hs

    @property
    def slopes(self):
        xs, _ = self.profile
        if self.__ms is None:
            return xs, self.__estimate_slopes()
        return xs, self.__ms

    def illumination(self, x, x_src, h_src, dissipation_power=2): # TODO: ONCE THIS WORKS, CLEAN UP THE IMPLEMENTATION
        def theta(m, dx, dy):
            return math.acos((dx + m * dy) / (math.sqrt(1 + m**2) * math.sqrt(dx**2 + dy**2)))

        h = self.h(x)

        if h > h_src:
            return 0
        elif x_src == x:
            return 1
        else:
            dissipation = pow(1 - abs(x_src - x), dissipation_power)
            m = self.m(x)
            on_left = (x_src > x)
            dx = (x_src - x) if on_left else (x - x_src)
            dy = (h_src - h) if on_left else (h - h_src)

            if on_left:
                if m > (dy / dx):
                    return 0
            else:
                if m < (dy / dx):
                    return 0
            return dissipation * math.sin(theta(m, dx, dy))
        return 0

    def debug_plot(self, fig=None, ax=None):
        if ax is None:
            fig, ax = plt.subplots(1, figsize=(16, 9), constrained_layout=True)
        ax.axis("off")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        for xs, hs in self.__profiles:
            ax.plot(xs, hs)
        return fig, ax

    def h(self, x, xs=None, hs=None):
        if xs is None:
            xs, hs = self.profile
        return self.__linear_interpolant(x, xs, hs, value_only=True)

    def m(self, x, xs=None, ms=None):
        if ms is None:
            xs, ms = self.slopes
        return self.__linear_interpolant(x, xs, ms, value_only=True)

    def snow_threshold(self, covered, xs=None, hs=None, max_iterations=5):
        if xs is None:
            xs, hs = self.profile
        iterations = 0
        h_min = 0
        h_max = np.max(hs)
        h = h_max / 2
        while iterations < max_iterations:
            count = (hs > h).sum()
            p = 100 * (count / len(hs))
            if p > covered:
                h_min = h
            else:
                h_max = h
            h = (h_min + h_max) / 2
            iterations += 1
        return h

    def __generate_profiles(self):
        self.__profiles = []
        num_peaks = self.__num_peaks
        ref_height = self.__ref_height
        for level in range(self.__levels):
            self.__profiles.append(self.__generate_single_profile(int(num_peaks), ref_height))
            num_peaks *= self.__frequency_multiplier
            ref_height *= self.__scale_multiplier
        return

    def __generate_single_profile(self, num_peaks, ref_height):
        h_peak_min, h_peak_max = 0.8*ref_height, ref_height
        h_valley_min, h_valley_max = 0.3*ref_height, 0.5*ref_height
        x = 0
        xs = [x]
        profile_mult = 1 if self.__profile_func is None else self.__profile_func(0)
        hs = [profile_mult * random.uniform(h_valley_min, h_valley_max)]

        dx_avg = 1 / (2*num_peaks + 1)
        for idx in range(num_peaks):
            x = self.__update_x(x, dx_avg)
            xs.append(x)
            profile_mult = 1 if self.__profile_func is None else self.__profile_func(x)
            hs.append(profile_mult * random.uniform(h_peak_min, h_peak_max))
            if x == 1:
                break
            x = 1 if (idx == num_peaks-1) else self.__update_x(x, dx_avg)
            xs.append(x)
            hs.append(profile_mult*random.uniform(h_valley_min, h_valley_max))
            if x == 1:
                break
        return (np.array(xs), np.array(hs))

    def __interpolate_profiles(self, profiles, max_level=None):
        max_level = len(profiles) if max_level is None else max_level
        xs_final = np.array(profiles[max_level-1][0])
        hs_final = np.array(profiles[max_level-1][1])
        for xs, hs in profiles[:max_level]:
            for idx, x in enumerate(xs_final):
                h = self.__linear_interpolant(x, xs, hs, value_only=True)
                hs_final[idx] += h
        self.__xs = xs_final
        self.__hs = hs_final
        return xs_final, hs_final

    def __estimate_slopes(self):
        xs_profile, hs_profile = self.profile
        ms = []
        bs = []
        for x in xs_profile:
            _, m, b = self.__linear_interpolant(x, xs_profile, hs_profile)
            ms.append(m)
            bs.append(b)
        self.__ms = np.array(ms)
        self.__bs = np.array(bs)
        return self.__ms

    def __linear_interpolant(self, x, xs, hs, value_only=False):
        m = 0
        b = 0
        if x <= 0:
            if value_only:
                return hs[0]
            m = (hs[1] - hs[0]) / xs[1]
            return hs[0], m, hs[0]
        if x >= 1:
            if value_only:
                return hs[-1]
            m = (hs[-1] - hs[-2]) / (xs[-1] - xs[-2])
            b = hs[-1] - (m * hs[-1])
            return hs[-1], m, b
        for idx, (xi, hi) in enumerate(zip(xs, hs)):
            if x == xi:
                if value_only:
                    return hi
                dx_pre = (xi - xs[idx-1])
                m_pre = (hi - hs[idx-1]) / dx_pre
                dx_post = (xs[idx+1] - xi)
                m_post = (hs[idx+1] - hi) / dx_post
                m = (dx_pre * m_pre + dx_post * m_post) / (dx_pre + dx_post)
                b = hi - (m * xi)
                return hi, m, b
            if xi > x:
                x1, h1 = xi, hi
                x0, h0 = xs[idx-1], hs[idx-1]
                m = (h1 - h0) / (x1 - x0)
                b = h1 - (m * x1)
                if value_only:
                    return (m * x) + b
                return (m * x) + b, m, b

    def __update_x(self, x, dx_avg, mult_min=0.7, mult_max=1.5):
        x_remaining = 1 - x
        dx = min(x_remaining, random.uniform(mult_min*dx_avg, mult_max*dx_avg))
        x += dx
        return x


__render_rand_state = None


def renderMountain(mountain, dims, base=None, mountain_color=(2, 6, 0, 255), fill_color=None, mode='RGBA', snow_threshold=0.2, snow_color=(48, 48, 56, 255), ill_kws={}):
    global __render_rand_state
    old_rand_state = random.getstate()
    if __render_rand_state is None:
        __render_rand_state = old_rand_state
    else:
        random.setstate(__render_rand_state)
    if fill_color is None:
        fill_color = (0, 0, 0, 0)
    if base is None:
        base = PIL.Image.new(mode, dims, color=fill_color)
    draw = PIL.ImageDraw.Draw(base)
    ill_color = None
    ill_strength = None
    render_illumination = False
    if 'x_src' in ill_kws and 'h_src' in ill_kws:
        ill_color = (192, 192, 128, 255) if 'color' not in ill_kws else ill_kws['color']
        ill_strength = 0.1 if 'strength' not in ill_kws else ill_kws['strength']
        render_illumination = True
    c_snow = snow_color
    c_mountain = mountain_color
    xs_profile, hs_profile = mountain.profile
    for x in range(dims[0]):
        x_val = x / dims[0]
        if render_illumination:
            base_illumination = mountain.illumination(x_val, ill_kws['x_src'], ill_kws['h_src'])
            strength = 1 - (ill_strength * base_illumination)
            c_snow = blend(snow_color, ill_color, strength)
            strength = 1 - (ill_strength * base_illumination / 3)
            c_mountain = blend(mountain_color, ill_color, strength)
        h = mountain.h(x_val, xs_profile, hs_profile)
        dy = dims[1] * h
        y = dims[1] - dy
        if h > snow_threshold:
            dh = 1.4 * (h - snow_threshold)
            dh = random.gauss(dh, dh/20)
            dy = dims[1] * dh
            draw.line(((x, y), (x, y + dy)), fill=c_snow)
            y += dy
        draw.line(((x, y), (x, dims[1])), fill=c_mountain)
    random.setstate(old_rand_state)
    return base


def renderMountainMask(mountain, dims):
    base = PIL.Image.new('1', dims, color=1)
    draw = PIL.ImageDraw.Draw(base)
    xs_profile, hs_profile = mountain.profile
    for x in range(dims[0]):
        x_val = x / dims[0]
        y = dims[1] - (dims[1] * mountain.h(x_val, xs_profile, hs_profile))
        draw.line(((x, y), (x, dims[1])), fill=0)
    return base


def drawIlluminator(img, ill_kws={}):
    if 'x_src' in ill_kws:
        ill_color = (192, 192, 128, 255) if ('color' not in ill_kws) else ill_kws['color']
        dims = img.size
        r = dims[1]/70
        print('illuminator radius = {}'.format(r))
        xc = dims[0] * ill_kws['x_src']
        yc = dims[1] - dims[1] * ill_kws['h_src']
        draw = PIL.ImageDraw.Draw(img)
        x0 = xc - r
        x1 = xc + r
        y0 = yc - r
        y1 = yc + r
        draw.ellipse(((x0, y0), (x1, y1)), fill=ill_color)
    return img


if __name__ == '__main__':
    dims = (1280, 720)
    ill_kws = {
        'x_src': 0.33,
        'h_src': 0.375,
        'strength': 0.25,
        'color': (200, 200, 160, 255)
    }
    random.seed(123456)
    mountain_profile = func_profiles.TriangleProfile(x1=(0.6, 0.7), y1=1, y_mults=(0.15, 0.3))
    mountain = MountainProfile(num_peaks=5, levels=5, scale_multiplier=0.25, profile_func=mountain_profile)
    fig, axs = plt.subplots(3, figsize=(16, 9), sharex=True, constrained_layout=True)
    fig, ax = mountain.debug_plot(ax=axs[0])
    xs, hs = mountain.profile
    h_snow = mountain.snow_threshold(27, xs, hs)
    axs[0].plot(xs, hs)
    axs[0].axhline(h_snow)
    axs[0].scatter([ill_kws['x_src']], [ill_kws['h_src']])
    xs, ms = mountain.slopes
    axs[1].plot(xs, ms)
    axs[1].axhline(0, c='k')
    ills = [mountain.illumination(x, ill_kws['x_src'], ill_kws['h_src']) for x in xs]
    axs[2].plot(xs, ills)
    plt.show()
    # xs, hs = mountain.illumination(0.5, 0.5)
    # img = PIL.Image.new('RGBA', dims, color=(0, 0, 0, 0))
    # img = drawIlluminator(img, ill_kws=ill_kws)
    # img = renderMountain(mountain, dims, snow_threshold=h_snow, ill_kws=ill_kws, base=img)
    # img.show()
    # img = PIL.Image.new('RGBA', dims, color=(0, 0, 0, 0))
    # img = drawIlluminator(img, ill_kws=ill_kws)
    # N = 10
    # img = img.filter(PIL.ImageFilter.MaxFilter(int(2*N + 1)))  # Dialate the illuminator - Odd Number
    # img = img.filter(PIL.ImageFilter.GaussianBlur(int(2.15 * N)))  # Blure the illuminator
    # img = drawIlluminator(img, ill_kws=ill_kws)
    # img.show()
    # fig, ax = plt.subplots(1, figsize=(16, 9), constrained_layout=True)
    # ax.plot(xs, hs)
    # ax.axis("off")
    # ax.set_xlim(0, 1)
    # ax.set_ylim(0, 1)

    # plt.show()
