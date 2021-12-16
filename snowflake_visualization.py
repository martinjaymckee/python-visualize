import datetime
import itertools
import math
import os
import os.path
import random

import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
import numpy as np
import PIL


def generateGrid(N, circular=True, highlight_drive=False, highlight_color='c', background_color='k'):
    def in_drive_section(c):
        return c[0] >= 0 and c[1] >= 0

    coord = []
    colors = []

    for c in itertools.product(range(-(N-1), N), repeat=2):
        pos = (c[0], c[1], -c[1])
        coord.append(pos)
        highlighted = highlight_drive and in_drive_section(pos)
        colors.append(highlight_color if highlighted else background_color)

    # Vertical cartesian coords
    vcoord = [c[0] for c in coord]

    # Horizontal cartersian coords
    hcoord = [(2. * np.sin(np.radians(60)) * (c[1] - c[2]) / 3.) + 0.5*c[0] for c in coord]

    if circular:
        new_coord = []
        new_colors = []
        new_hcoord = []
        new_vcoord = []
        for c, color, x, y in zip(coord, colors, hcoord, vcoord):
            d = math.sqrt(x**2 + y**2)
            if d <= N:
                new_coord.append(c)
                new_colors.append(color)
                new_hcoord.append(x)
                new_vcoord.append(y)
        return new_coord, new_colors, new_hcoord, new_vcoord

    return coord, colors, hcoord, vcoord


def generateRandomSnowflake(coord, colors, branch_color='c', ray_color='g'):
    def genBranch(u0, v0, N):
        continue_probability = 0.8
        branch_probability = 0.15 #0.05 #0.15
        p = random.uniform(0, 1)
        setColorSymmetric(u0, v0, branch_color)
        if continue_probability > p:
            if u0 < N-1:
                genBranch(u0+1, v0, N)
        if branch_probability > p:
            if v0 < N-1:
                genBranch(u0, v0+1, N)
        return

    def genRay(N):
        start_probability = 0.65
        end_probability = 0.1
        branch_probability = 0.3
        max_ring = int(N/4)
        min_size = int(N/2)
        started = False
        start_idx = None
        for v in range(N):
            p = random.uniform(0, 1)
            if not started:
                if p < start_probability or (v == max_ring):
                    started = True
                    start_idx = v
                    # Ensure Connected
                    if v > 1:
                        for v_off in range(v):
                            setColorSymmetric(v_off, v-v_off, ray_color)

            if started:
                setColorSymmetric(0, v, ray_color)
                if p < branch_probability:
                    genBranch(1, v, N)
                if (v >= min_size) and (p < end_probability):
                    return start_idx
        return start_idx

    def setColor(u, v, color, debug=False):
        coords = [
            (u, v, -v),
            (-u, -v, v),
            (u, -v-u, v+u),
            (-u, v+u, -v-u),
            (u+v, -u, u),
            (-(u+v), u, -u)
        ]
        for c in coords:
            try:
                idx = coord.index(c)
                colors[idx] = color
            except Exception:
                if debug:
                    print('index ({}, {}, {}) not found'.format(u, v, -v))

    def setColorSymmetric(u, v, color):
        setColor(u, v, color)
        setColor(v, u, color)

    N = max([c[0] for c in coord]) + 1
    start_v = genRay(N)
    return colors, (0, start_v, -start_v)


def plotHex(ax, hcoord, vcoord, colors, labels=None, ignore_color=None):
    ax.set_aspect('equal')
    ax.set_facecolor((0, 0, 0))
    # Add some coloured hexagons
    for x, y, color in zip(hcoord, vcoord, colors):
        if color is not ignore_color:
            hex = RegularPolygon((x, y), numVertices=6, radius=2. / 3.,
                                 facecolor=color, alpha=0.8, edgecolor=None)
            ax.add_patch(hex)

    # Also add scatter points in hexagon centres
    # ax.scatter(hcoord, vcoord, c=[c[0].lower() for c in colors], alpha=0.5)

    if labels is not None:
        for x, y, l in zip(hcoord, vcoord, labels):
            ax.text(x, y, l, ha='center', va='center', size=6)

    ax.set_xticks([], minor=[])
    ax.set_yticks([], minor=[])
    ax.autoscale_view()


def generateSnowflakeDirectory(directory, num, N=150, resolution=2000):
    c_transparent = (0, 0, 0, 0)
    if not os.path.exists(directory):
        os.makedirs(directory)
    for idx in range(num):
        filename = 'snowflake_{}.png'.format(idx)
        fig, ax = plt.subplots(1, figsize=(1, 1), constrained_layout=True, dpi=resolution)
        fig.set_facecolor(c_transparent)
        ax.set_facecolor(c_transparent)
        ax.axis("off")
        print('Creating {} of {} -> {}...'.format(idx+1, num, filename))
        coord, colors, hcoord, vcoord = generateGrid(N, background_color=c_transparent)
        colors, c_start = generateRandomSnowflake(coord, colors, branch_color='w', ray_color='w')
        plotHex(ax, hcoord, vcoord, colors, ignore_color=c_transparent)
        fig.savefig(os.path.join(directory, filename), transparent=True)


class SnowflakePlane:
    def __init__(self, dims, angle, distance, src_snowflakes, scale=0.05, density=0.4, velocity=0.15, velocity_sd=0.025, rotation_sd=0.45):
        self.__dims = dims
        self.__angle = angle
        self.__distance = distance
        self.__src_snowflakes = src_snowflakes
        image_scale = 1 / distance  # TODO: FIGURE OUT IF THESE ARE CORRECT WAYS TO CALCULATE....
        self.__snowflake_size = scale * image_scale  # TODO: Calculate Size of Snowflakes at plane distance
        velocity_scale = 1 / distance
        self.__velocity = velocity * velocity_scale  # TODO: Calculate actual mean velocity and standard deviation
        self.__velocity_sd = velocity_sd * velocity_scale
        self.__rotation_sd = rotation_sd
        self.__num_snowflakes = int(math.ceil(density / (self.__snowflake_size**2)))
        self.__snowflakes = []
        print('\tInitializing {} snowflakes at {} pixels'.format(self.__num_snowflakes, self.__snowflake_size * dims[1]))
        start = datetime.datetime.now()
        for idx in range(self.__num_snowflakes):
            self.__snowflakes.append(self.__new_snowflake(initialize=True))
        delta = datetime.datetime.now() - start
        print('\t\tdelta_t = {}'.format(delta))

    @property
    def dims(self):
        return self.__dims

    @property
    def distance(self):
        return self.__distance

    @property
    def snowflakes(self):
        snowflakes = []
        for (x, y, alpha, theta), _, _, _, img in self.__snowflakes:
            x = int(x * self.__dims[0])
            y = int(y * self.__dims[1])
            snowflakes.append(((x, y, alpha, theta), img))
        return snowflakes

    # @profile
    def update(self, t, dt, wind_func=None):
        snowflakes = []
        z_limit = 1 + self.__snowflake_size
        for pos, vz, omega_a, omega_b, img in self.__snowflakes:
            dz = dt * vz
            pos[1] += dz
            if wind_func is not None:
                h = (pos[1] - 0.5) * self.__distance
                dx = dt * wind_func(t, h, self.__distance) / self.__distance
                pos[0] += dx
            pos[2] += dt * omega_a
            pos[3] += dt * omega_b
            if pos[1] > z_limit:
                snowflakes.append(self.__new_snowflake())
            else:
                snowflakes.append((pos, vz, omega_a, omega_b, img))
        self.__snowflakes = snowflakes


    def __new_snowflake(self, initialize=False, copy_probability=0.95):
        pixels = int(min(*self.__dims) * self.__snowflake_size)
        count_limit = 0.75 * pixels ** 2
        two_pi = 2 * math.pi
        x = random.uniform(0, 1)
        y = random.uniform(-self.__snowflake_size, 1 if initialize else 0)
        alpha = random.uniform(-two_pi, two_pi)
        theta = random.uniform(-two_pi, two_pi)
        pos = [x, y, alpha, theta]
        vz = abs(random.gauss(self.__velocity, self.__velocity_sd)) + (self.__velocity_sd / 10)
        omega_a = random.gauss(0, self.__rotation_sd)
        omega_b = random.gauss(0, self.__rotation_sd)
        copy_created = (len(self.__snowflakes) >= count_limit) and (random.uniform(0, 1) < copy_probability)
        img = random.choice(self.__snowflakes)[4] if copy_created else self.__load_random_snowflake_image(pixels, pixels)
        return (pos, vz, omega_a, omega_b, img)

    def __load_random_snowflake_image(self, w, h):
        img = random.choice(self.__src_snowflakes)
        img = img.resize((w, h))
        return img


class SnowflakeScene:
    @classmethod
    def FromDistances(cls, dims, angle, distances, src_snowflakes, plane_kwargs={}, **kwargs):
        layers = []
        for distance in sorted(distances, reverse=True):
            print('Generating snowflake layer at {}'.format(distance))
            layer = SnowflakePlane(dims, angle, distance, src_snowflakes, **plane_kwargs)
            layers.append(layer)
        return SnowflakeScene(layers, **kwargs)

    def __init__(self, layers, background_color=(0, 0, 32, 255), atmospheric_color=(85, 85, 128, 128), max_distance=None):
        self.__layers = layers
        self.__background_color = background_color
        self.__atmospheric_color = atmospheric_color
        self.__max_distance = 1.5 * (max(*[layer.distance for layer in layers]) if max_distance is None else max_distance)
        self.__t = 0

    def image(self, base_img=None, atmos_img=None):
        dims = self.__layers[0].dims
        if base_img is None:
            base_img = PIL.Image.new('RGBA', dims, color=self.__background_color)
        if atmos_img is None:
            atmos_img = PIL.Image.new('RGBA', dims, color=self.__atmospheric_color)
        for idx, layer in enumerate(self.__layers):
            for (x, y, alpha, theta), img in layer.snowflakes:
                # print('x = {}, y = {}, alpha = {}, theta = {}, img = {}'.format(x, y, alpha, theta, img))
                w, h = img.size
                squish_scale = abs(math.cos(alpha))
                img = img.resize((int(max(1, w * squish_scale + 0.5)), h))
                img = img.rotate(math.degrees(theta), expand=True)
                w, h = img.size
                base_img.paste(img, (int(x-w/2), int(y-h/2)), mask=img)
            if idx < len(self.__layers) - 1:
                base_img = PIL.Image.blend(base_img, atmos_img, 0.7 * pow(layer.distance / self.__max_distance, 2.5))
        return base_img

    def update(self, dt, wind_func=None):
        self.__t += dt
        for layer in self.__layers:
            layer.update(self.__t, dt, wind_func)


def vertical_gradient(c_bottom, c_top, dims, mode='RGB'):
    """Generate a vertical gradient."""
    width, height = dims
    base = PIL.Image.new(mode, (width, height), c_top)
    top = PIL.Image.new(mode, (width, height), c_bottom)
    mask = PIL.Image.new('L', (width, height))
    mask_data = []
    for y in range(height):
        mask_data.extend([int(255 * (y / height))] * width)
    mask.putdata(mask_data)
    base.paste(top, (0, 0), mask)
    return base


def distance_list(z0, zn, N):
    C = math.exp((1/(N-1))*(math.log(zn / z0)))
    return [z0 * pow(C, i) for i in range(N)]


if __name__ == "__main__":
    snowflake_directory = 'snowflake_images'
    num_snowflakes = 100
    snowflake_resolution = 1000
    snowflake_diameter_cells = 150
    generateSnowflakeDirectory(snowflake_directory, num_snowflakes, N=snowflake_diameter_cells, resolution=snowflake_resolution)
    #
    # src_snowflakes = []
    # for file in os.listdir(snowflake_directory):
    #     if file.endswith(".png"):
    #         path = os.path.join(snowflake_directory, file)
    #         # print('Loading {}'.format(path))
    #         img = PIL.Image.open(path)
    #         src_snowflakes.append(img)
    # print('{} snowflakes loaded'.format(len(src_snowflakes)))
    #
    # def simple_wind(t, h, d):
    #     time_scale = 0.1
    #     v_max = 0.035
    #     v_sd = 0.001
    #     return random.gauss(v_max * math.sin(time_scale*t + h + d), v_sd)
    #
    # scene = SnowflakeScene.FromDistances(dims, math.radians(50), [0.25, 0.44, 0.77, 1.34, 2.34, 4.10, 7.18, 12.57], src_snowflakes)
    #
    # dt = 1 / fps
    #
    # for _ in range(10):
    #     t_start = datetime.datetime.now()
    #     scene.update(dt, wind_func=simple_wind)
    #     print('t = {}'.format(datetime.datetime.now() - t_start))

    # # scene = SnowflakeScene.FromDistances(dims, math.radians(50), [0.25, 0.44, 0.77, 1.34, 2.34, 4.10], src_snowflakes)
    # # scene = SnowflakeScene.FromDistances(dims, math.radians(50), [0.3, 0.45, 0.68, 1.01, 1.52, 2.28, 3.42, 5.13], src_snowflakes)
    # # scene = SnowflakeScene.FromDistances(dims, math.radians(50), [0.35, 0.63, 1.13, 2.04, 3.67, 6.61], src_snowflakes)
    # scene = SnowflakeScene.FromDistances(dims, math.radians(50), [0.35, 0.63, 1.13], src_snowflakes)
    #

    # N = int(t_play * fps)
    #
    # print('Deleting previous frames')
    # for file in os.listdir(frame_directory):
    #     os.remove(os.path.join(frame_directory, file))
    #
    # print('Creating new frames')
    # for idx in range(N):
    #     filename = 'frame_{:05d}.png'.format(idx)
    #     print('\tGenerating {}, {} of {}'.format(filename, idx+1, N))
    #     frame = scene.image(base_img=background_img.copy())
    #     frame.save(os.path.join(frame_directory, filename))
    #     scene.update(dt, simple_wind)
    #
    # print('Generate Animation...')
    # clip = mpy.ImageSequenceClip(frame_directory, fps=fps)
    # animation_path = os.path.join(output_directory, 'test_animation.mp4')
    # clip.write_videofile(animation_path, fps=fps)
    # print('Animation saved as {}'.format(animation_path))
    #
    # # generateSnowflakeDirectory('snowflake_images', 2500)
    #
    # # rows = 5
    # # columns = 5
    # # N = 40
    # # debug = False
    # # fig, axs = plt.subplots(rows, columns, constrained_layout=True)
    # # fig.set_facecolor('k')
    # #
    # # for row in range(rows):
    # #     for column in range(columns):
    # #         coord, colors, hcoord, vcoord = generateGrid(N)
    # #         labels = [str(c) for c in coord] if debug else None
    # #         colors, c_start = generateRandomSnowflake(coord, colors, branch_color='w', ray_color='w')
    # #         plotHex(axs[row, column], hcoord, vcoord, colors, labels=labels)
    # #
    # # plt.show()
