import math
import random

import numpy as np


# TODO: IMPROVE SLOPE ESTIMATES IN CYCLIC CASE

class FractalProfile:
    @classmethod
    def Circular(cls, **kwargs):
        return FractalProfile(x_lims=(-math.pi, math.pi), cyclic=True, **kwargs)

    def __init__(self, ref_height=1, offset_height=0, num_peaks=10, levels=5, frequency_multiplier=2.5, scale_multiplier=0.5, peak_scale=(0.8, 1.0), valley_scale=(0.3, 0.5), x_lims=(0, 1), cyclic=False, profile_func=None):
        self.__cyclic = cyclic
        self.__num_peaks = num_peaks
        self.__ref_height = ref_height
        self.__offset_height = offset_height
        self.__levels = levels
        self.__frequency_multiplier = frequency_multiplier
        self.__scale_multiplier = scale_multiplier
        self.__peak_scale = peak_scale
        self.__valley_scale = valley_scale
        self.__profile_func = profile_func
        self.__profiles = []
        self.__x_lims = x_lims
        self.__generate_profiles()
        self.__xs = None
        self.__hs = None
        self.__ms = None
        self.__bs = None
        self.__map_m = (self.__x_lims[1] - self.__x_lims[0])
        self.__map_b = self.__x_lims[1] - self.__map_m

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

    def extents(self, samples=100):
        sample_xs = np.linspace(*self.__x_lims, samples)
        hs = np.array([self.params(x)[0] for x in sample_xs])
        return np.min(hs), np.max(hs)

    def params(self, x):
        x1 = self.__map_x(x)
        # print('x = {} -> x1 = {}'.format(x, x1))
        if self.__xs is None:
            self.__interpolate_profiles(self.__profiles)
        return self.__linear_interpolant(x1, self.__xs, self.__hs)

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
        h_peak_min, h_peak_max = self.__peak_scale[0]*ref_height, self.__peak_scale[1]*ref_height
        h_valley_min, h_valley_max = self.__valley_scale[0]*ref_height, self.__valley_scale[1]*ref_height
        x = 0
        xs = [x]
        profile_mult = 1 if self.__profile_func is None else self.__profile_func(0)
        hs = [profile_mult * random.uniform(h_valley_min, h_valley_max)]

        dx_avg = 1 / (2*num_peaks + 1)
        for idx in range(num_peaks):
            x = self.__update_x(x, dx_avg)
            if (x == 1) and (idx == num_peaks - 1) and self.__cyclic:
                xs.append(1)
                hs.append(hs[0])
            xs.append(x)
            profile_mult = 1 if self.__profile_func is None else self.__profile_func(x)
            hs.append(profile_mult * random.uniform(h_peak_min, h_peak_max))
            if x == 1:
                break
            x = 1 if (idx == num_peaks-1) else self.__update_x(x, dx_avg)
            if (x == 1) and self.__cyclic:
                xs.append(1)
                hs.append(hs[0])
                break
            else:
                xs.append(x)
                hs.append(profile_mult*random.uniform(h_valley_min, h_valley_max))
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
            return self.__offset_height + hs[0], m, hs[0]
        if x >= 1:
            if value_only:
                return hs[-1]
            m = (hs[-1] - hs[-2]) / (xs[-1] - xs[-2])
            b = hs[-1] - (m * hs[-1])
            return self.__offset_height + hs[-1], m, b
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
                return self.__offset_height + hi, m, b
            if xi > x:
                x1, h1 = xi, hi
                x0, h0 = xs[idx-1], hs[idx-1]
                m = (h1 - h0) / (x1 - x0)
                b = h1 - (m * x1)
                if value_only:
                    return (m * x) + b
                #  TODO: FIX THE SLOPE AND INTERCEPT CALCULATIONS TO CORRECTLY HANDLE THE OFFSET???
                return self.__offset_height + (m * x) + b, m, b

    def __map_x(self, x):
        return (x - self.__map_b) / self.__map_m

    def __update_x(self, x, dx_avg, mult_min=0.7, mult_max=1.5):
        x_remaining = 1 - x
        dx = min(x_remaining, random.uniform(mult_min*dx_avg, mult_max*dx_avg))
        x += dx
        return x


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    profile = FractalProfile.Circular(ref_height=0.25, offset_height=3.5, levels=5)

    xs = np.linspace(-math.pi, math.pi, 10000)
    ys = [profile.params(x)[0] for x in xs]

    plt.plot(xs, ys, alpha=0.5)
    plt.axhline(ys[0], alpha=0.25)
    print('profile.extents() = {}'.format(profile.extents()))
    plt.show()
