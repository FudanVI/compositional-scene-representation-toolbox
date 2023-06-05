import argparse
import gzip
import json
import os
import pdb
import shutil

import bpy
import numpy as np

import utils


def generate_mask(path, blender_objects, channel_split=3, antialiasing=True):
    render_args = bpy.context.scene.render
    # Cache the render args we are about to clobber
    old_filepath = render_args.filepath
    old_engine = render_args.engine
    old_filter_size = render_args.filter_size
    old_film_transparent = render_args.film_transparent
    old_color_mode = render_args.image_settings.color_mode
    # Override some render settings to have flat shading
    render_args.filepath = path
    render_args.engine = 'CYCLES'
    render_args.filter_size = 0.0
    render_args.film_transparent = True
    render_args.image_settings.color_mode = 'RGBA'
    # Switch denoising state
    old_denoising_state = bpy.context.scene.node_tree.nodes["Switch"].check
    bpy.context.scene.node_tree.nodes["Switch"].check = False
    # Don't render lights
    utils.set_render(bpy.data.objects['Lamp_Key'], False)
    utils.set_render(bpy.data.objects['Lamp_Fill'], False)
    utils.set_render(bpy.data.objects['Lamp_Back'], False)
    utils.set_render(bpy.data.objects['Ground'], False)
    # Change shading and AA
    old_shading = bpy.context.scene.display.shading.light
    bpy.context.scene.display.shading.light = 'FLAT'
    old_aa = bpy.context.scene.display.render_aa
    bpy.context.scene.display.render_aa = 'OFF'
    # Add random shadeless materials to all objects
    new_obj = []
    for obj in bpy.data.objects:
        obj.select_set(state=False)
    vis_id = []
    for i, obj in enumerate(blender_objects):
        if obj.hide_render:
            continue
        vis_id.append(i)
        obj.select_set(state=True)
        bpy.ops.object.duplicate(linked=False, mode='INIT')
        utils.set_render(obj, False)
        bpy.ops.material.new()
        mat = bpy.data.materials['Material']
        mat.name = 'Material_temp_%d' % i
        r = (i % channel_split) / (channel_split - 1)
        idx = i // channel_split
        g = (idx % channel_split) / (channel_split - 1)
        idx //= channel_split
        b = (idx % channel_split) / (channel_split - 1)
        mat.diffuse_color = [r, g, b, 1]
        mat.shadow_method = 'NONE'
        new_obj.append(bpy.context.selected_objects[0])
        for i in range(len(bpy.context.selected_objects[0].data.materials)):
            bpy.context.selected_objects[0].data.materials[i] = mat
        for o in bpy.data.objects:
            o.select_set(state=False)
    # Render the scene
    # Save gamma

    gamma = bpy.context.scene.view_settings.view_transform
    bpy.context.scene.view_settings.view_transform = 'Raw'
    bpy.ops.render.render(write_still=True)
    bpy.context.scene.view_settings.view_transform = gamma
    # Undo the above; first restore the materials to objects
    for obj in new_obj:
        obj.select_set(state=True)
        bpy.ops.object.delete()
    for i in vis_id:
        utils.set_render(blender_objects[i], True)
    # Render lights again
    utils.set_render(bpy.data.objects['Lamp_Key'], True)
    utils.set_render(bpy.data.objects['Lamp_Fill'], True)
    utils.set_render(bpy.data.objects['Lamp_Back'], True)
    utils.set_render(bpy.data.objects['Ground'], True)
    # Set the render settings back to what they were
    render_args.filepath = old_filepath
    render_args.engine = old_engine
    render_args.filter_size = old_filter_size
    render_args.film_transparent = old_film_transparent
    render_args.image_settings.color_mode = old_color_mode
    bpy.context.scene.display.shading.light = old_shading
    bpy.context.scene.display.render_aa = old_aa
    bpy.context.scene.node_tree.nodes["Switch"].check = old_denoising_state
    return


def generate_images(folder, blender_objects):
    if args.use_gpu == 1:
        # Blender changed the API for enabling CUDA at some point
        if bpy.app.version < (2, 78, 0):
            bpy.context.user_preferences.system.compute_device_type = 'CUDA'
            bpy.context.user_preferences.system.compute_device = 'CUDA_0'
        elif bpy.app.version < (2, 80, 0):
            cycles_prefs = bpy.context.user_preferences.addons['cycles'].preferences
            cycles_prefs.compute_device_type = 'CUDA'
        else:
            cycles_prefs = bpy.context.preferences.addons['cycles'].preferences
            cycles_prefs.compute_device_type = 'CUDA'
    bpy.data.worlds['World'].cycles.sample_as_light = True
    # Some CYCLES-specific stuff
    bpy.context.scene.cycles.blur_glossy = 2.0
    bpy.context.scene.cycles.samples = args.render_num_samples
    bpy.context.scene.cycles.transparent_min_bounces = args.render_min_bounces
    bpy.context.scene.cycles.transparent_max_bounces = args.render_max_bounces
    if args.use_gpu:
        bpy.context.scene.cycles.device = 'GPU'
    bpy.context.preferences.addons['cycles'].preferences.get_devices()
    render_args = bpy.context.scene.render
    old_filepath = render_args.filepath
    ground = bpy.data.objects['Ground']
    ground.location[0] = 30
    ground.location[1] = 84
    ground.location[2] = 0
    ground.rotation_euler[0] = ground.rotation_euler[1] = ground.rotation_euler[2] = 0
    ground.scale[0] = 3
    ground.scale[1] = 3
    ground.scale[2] = 0
    delta_theta = (args.max_theta - args.min_theta) / args.num_views
    phi = 0.5 * np.pi * np.random.uniform(args.min_phi, args.max_phi)
    rho = np.random.uniform(args.min_rho, args.max_rho)
    for idx in range(args.num_views):
        theta = 2 * np.pi * (args.min_theta + delta_theta * idx)
        # theta = 2 * np.pi * np.random.uniform(args.min_theta, args.max_theta)
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)
        bpy.context.scene.camera.location[0] = rho * cos_phi * cos_theta
        bpy.context.scene.camera.location[1] = rho * cos_phi * sin_theta
        bpy.context.scene.camera.location[2] = rho * sin_phi
        with open(os.path.join(folder, 'view_{}.json'.format(idx)), 'w') as f:
            json.dump({'theta': theta, 'phi': phi, 'rho': rho}, f)
        # Scene
        render_args.filepath = os.path.join(folder, 'image_{}.png'.format(idx))
        bpy.ops.render.render(write_still=True)
        # Segmentation
        generate_mask(os.path.join(folder, 'segment_{}.png'.format(idx)), blender_objects)
        # Background

        for i, obj in enumerate(blender_objects):
            utils.set_render(obj, False)
        #     render_args.filepath = os.path.join(folder, 'back.png')
        #     bpy.ops.render.render(write_still=True)
        # Objects
        for i, obj in enumerate(blender_objects):
            utils.set_render(obj, True)
            #         render_args.filepath = os.path.join(folder, 'object_{}.png'.format(i))
            #         bpy.ops.render.render(write_still=True)
            generate_mask(os.path.join(folder, 'mask_{}_{}.png'.format(idx, i)), blender_objects)
            utils.set_render(obj, False)
        for i, obj in enumerate(blender_objects):
            utils.set_render(obj, True)
    render_args.filepath = old_filepath
    return


def main():
    for idx in range(args.offset, args.offset + args.num_images):
        np.random.seed(idx)
        filename_zip = 'SHOP_VRB_train_{:06d}.blend.gz'.format(idx)
        filename_unzip = '.'.join(filename_zip.split('.')[:-1])
        path_zip = os.path.join(args.folder_blend, filename_zip)
        path_unzip = os.path.join(args.folder_blend, filename_unzip)
        with gzip.open(path_zip, 'rb') as f_in, open(path_unzip, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
        bpy.ops.wm.open_mainfile(filepath=path_unzip)
        os.remove(path_unzip)
        name_data = '.'.join(filename_unzip.split('.')[:-1])
        folder_out = os.path.join(args.folder_image, name_data)
        if not os.path.exists(folder_out):
            os.makedirs(folder_out)
        with open(os.path.join(args.folder_scene, name_data + '.json'), 'r') as f:
            scene = json.load(f)
        blender_objects = [bpy.data.objects[n['name_raw']] for n in scene['objects']]
        generate_images(folder_out, blender_objects)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder_image')
    parser.add_argument('--folder_scene')
    parser.add_argument('--folder_blend')
    parser.add_argument('--offset', type=int)
    parser.add_argument('--num_images', type=int)
    parser.add_argument('--num_views', type=int)
    parser.add_argument('--min_theta', type=float)
    parser.add_argument('--max_theta', type=float)
    parser.add_argument('--min_phi', type=float)
    parser.add_argument('--max_phi', type=float)
    parser.add_argument('--min_rho', type=float)
    parser.add_argument('--max_rho', type=float)
    parser.add_argument('--use_gpu', type=int, default=1)
    parser.add_argument('--render_num_samples', default=64, type=int)
    parser.add_argument('--render_min_bounces', default=8, type=int)
    parser.add_argument('--render_max_bounces', default=8, type=int)
    argv = utils.extract_args()
    args = parser.parse_args(argv)
    main()
