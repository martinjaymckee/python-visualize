import math
import random


def default_wind(t, h, d):
    time_scale = 0.1
    v_max = 0.035
    v_sd = 0.001
    return random.gauss(v_max * math.sin(time_scale*t + h + d), v_sd)


if __name__ == '__main__':
    import datetime
    import os
    import os.path

    import blend_modes
    import moviepy.editor as mpy
    import numpy as np
    import PIL

    import func_profiles
    import moon_render
    import mountain_range
    import random_snowflake_plotting_test

    #
    # Configuration Values
    #
    # random.seed(123456)

    # fps = 24
    # fps = 30
    fps = 60
    # t_play = 300
    # dims = (1280, 720)
    # dims = (1920, 1080)
    # dims = (640, 360)
    t_play = 3 * 60 + 45
    dims = (3840, 2160)

    snowflake_directory = 'snowflake_images'
    frame_directory = 'snowy_landscape_frames'
    moon_frame_directory = 'moon_reference_frames'
    output_directory = 'snowy_landscape_outputs'

    atmosphere_color = (30, 30, 40, 16)  # (85, 85, 128, 128)
    sky_bottom_color = (8, 8, 48, 255)
    sky_top_color = (0, 0, 0, 255)
    illumination_strength = 0.55

    z_snow_min = 0.25
    z_snow_max = 20
    N_snow_layers = 10

    view_width = math.radians(30)

    #
    # Processing code
    #
    dt = 1 / fps

    view = moon_render.ViewFromDimsAndAngularWidth(dims, view_width)

    src_snowflakes = []
    print('Load Snowflake Images')
    for file in os.listdir(snowflake_directory):
        if file.endswith(".png"):
            path = os.path.join(snowflake_directory, file)
            # print('Loading {}'.format(path))
            img = PIL.Image.open(path)
            src_snowflakes.append(img)
    print('\t{} snowflakes loaded'.format(len(src_snowflakes)))

    print('Generate Snow Scene')
    snow_distances = random_snowflake_plotting_test.distance_list(z_snow_min, z_snow_max, N_snow_layers)
    snow_scene = random_snowflake_plotting_test.SnowflakeScene.FromDistances(dims, view[1], snow_distances, src_snowflakes)

    print('Generate Base Images (Background and Atmosphere)')
    background_img = random_snowflake_plotting_test.vertical_gradient(sky_bottom_color, sky_top_color, dims, mode='RGBA')
    atmosphere_img = PIL.Image.new('RGBA', dims, color=atmosphere_color)

    print('Generate Empty Image')
    clear_img = PIL.Image.new('RGBA', dims, color=(0, 0, 0, 0))

    print('Generate Mountain Profile')
    mountain_profile = func_profiles.TriangleProfile(x1=(0.6, 0.7), y1=1, y_mults=(0.15, 0.3))
    mountain = mountain_range.MountainProfile(num_peaks=5, levels=7, scale_multiplier=0.3, profile_func=mountain_profile)
    xs, hs = mountain.profile
    print('Generate Mountain Mask')
    mountain_mask = mountain_range.renderMountainMask(mountain, dims)
    h_snow = mountain.snow_threshold(29, xs, hs)

    print('\t\tGenerate Moon')  # TODO: THIS SHOULD ACTUALLY BE REIMPLEMENTED WITH AN UPDATE ON THE MOON OBJECT
    h_moon_final = random.uniform(0.8, 1.0)
    idx = int(len(xs) / 3)
    x_moon_init, h_moon_init = xs[idx], 0.75*hs[idx]
    moon = moon_render.Moon.SizeFromView(x_moon_init, h_moon_init, dims[0], view, percent=20)
    moon.calculate_mult_t(h_moon_final, t_play)
    h_offset = moon.img.size[0] / (2 * dims[1])

    num_frames = int(fps * t_play + 0.5)

    print('Deleting previous frames')
    for file in os.listdir(frame_directory):
        os.remove(os.path.join(frame_directory, file))

    update_landscape = True
    landscape = None
    atmosphere = atmosphere_img.copy()
    print('Generating {} Frames'.format(num_frames))
    for idx in range(num_frames):
        print('\tGenerate Frame {}'.format(idx))
        t_start_frame = datetime.datetime.now()
        # print('Create Background')
        if (landscape is None) or update_landscape:
            print('\t\tUpdate Landscape')
            landscape = background_img.copy()
            print('\t\t\tRender Moon')
            landscape = moon.render(landscape)
            glow = None
            if (moon.pos[1]+h_offset) > mountain.h(moon.pos[0]):
                print('\t\t\tRender Glow')
                glow = moon.glow(dims, mask=mountain_mask)
            print('\t\t\tRender Mountains')
            ill_kws = moon.ill_kws
            ill_kws['strength'] = illumination_strength
            mid = mountain_range.renderMountain(mountain, dims, snow_threshold=h_snow, ill_kws=ill_kws)
            landscape = PIL.Image.alpha_composite(landscape, mid)
            if glow is None:
                atmosphere = atmosphere_img.copy()
            else:
                atmosphere_img_float = np.array(atmosphere_img.copy()).astype(float)
                glow_img_float = np.array(glow).astype(float)
                blended_img_float = blend_modes.lighten_only(atmosphere_img_float, glow_img_float, 1.0)
                atmosphere = PIL.Image.fromarray(np.uint8(blended_img_float))
        print('\t\tRender Snow')
        frame = snow_scene.image(base_img=landscape.copy(), atmos_img=atmosphere)
        print('\t\tUpdate Moon')
        update_landscape = moon.update(dt)
        moon_frame = moon.render(clear_img.copy(), mask=mountain_mask)
        filename = 'frame_{:05d}.png'.format(idx)
        moon_frame.save(os.path.join(moon_frame_directory, filename))
        print('\t\tUpdate Snow Scene')
        t_start = datetime.datetime.now()
        snow_scene.update(dt, default_wind)
        print('\t\t\tt = {}'.format(datetime.datetime.now() - t_start))
        print('\t\tSave frame -- {}'.format(filename))
        frame.save(os.path.join(frame_directory, filename))
        print('\t\tFrame Time = {}'.format(datetime.datetime.now() - t_start_frame))
    print('Generate Moon Reference Animation...')
    clip = mpy.ImageSequenceClip(moon_frame_directory, fps=fps)
    animation_path = os.path.join(output_directory, 'test_moon_reference_animation.mp4')
    clip.write_videofile(animation_path, fps=fps)
    print('Animation saved as {}'.format(animation_path))

    print('Generate Main Animation...')
    clip = mpy.ImageSequenceClip(frame_directory, fps=fps)
    animation_path = os.path.join(output_directory, 'test_animation.mp4')
    clip.write_videofile(animation_path, fps=fps)
    print('Animation saved as {}'.format(animation_path))
