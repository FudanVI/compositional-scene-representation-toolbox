import argparse
import gzip
import json
import os
import shutil

import bpy
import numpy as np

import utils


def generate_mask(path, blender_objects, channel_split=3, antialiasing=True):
    render_args = bpy.context.scene.render
    # Cache the render args we are about to clobber
    old_filepath = render_args.filepath
    old_engine = render_args.engine
    old_use_antialiasing = render_args.use_antialiasing
    old_alpha_mode = render_args.alpha_mode
    old_color_mode = render_args.image_settings.color_mode
    # Override some render settings to have flat shading
    render_args.filepath = path
    render_args.engine = 'BLENDER_RENDER'
    render_args.use_antialiasing = antialiasing
    render_args.alpha_mode = 'TRANSPARENT'
    render_args.image_settings.color_mode = 'RGBA'
    # Move the lights and ground to layer 2 so they don't render
    utils.set_layer(bpy.data.objects['Lamp_Key'], 2)
    utils.set_layer(bpy.data.objects['Lamp_Fill'], 2)
    utils.set_layer(bpy.data.objects['Lamp_Back'], 2)
    utils.set_layer(bpy.data.objects['Ground'], 2)
    utils.set_layer(bpy.data.objects['Ground_extra'], 2)
    # Add random shadeless materials to all objects
    old_materials = []
    for i, obj in enumerate(blender_objects):
        old_materials.append(obj.data.materials[0])
        bpy.ops.material.new()
        mat = bpy.data.materials['Material']
        mat.name = 'Material_%d' % i
        r = (i % channel_split) / (channel_split - 1)
        idx = i // channel_split
        g = (idx % channel_split) / (channel_split - 1)
        idx //= channel_split
        b = (idx % channel_split) / (channel_split - 1)
        mat.diffuse_color = [r, g, b]
        mat.use_shadeless = True
        obj.data.materials[0] = mat
    # Render the scene
    bpy.ops.render.render(write_still=True)
    # Undo the above; first restore the materials to objects
    for mat, obj in zip(old_materials, blender_objects):
        obj.data.materials[0] = mat
    # Move the lights and ground back to layer 0
    utils.set_layer(bpy.data.objects['Lamp_Key'], 0)
    utils.set_layer(bpy.data.objects['Lamp_Fill'], 0)
    utils.set_layer(bpy.data.objects['Lamp_Back'], 0)
    utils.set_layer(bpy.data.objects['Ground'], 0)
    utils.set_layer(bpy.data.objects['Ground_extra'], 0)
    # Set the render settings back to what they were
    render_args.filepath = old_filepath
    render_args.engine = old_engine
    render_args.use_antialiasing = old_use_antialiasing
    render_args.alpha_mode = old_alpha_mode
    render_args.image_settings.color_mode = old_color_mode
    return


def generate_images(folder, blender_objects):
    if args.use_gpu == 1:
        # Blender changed the API for enabling CUDA at some point
        if bpy.app.version < (2, 78, 0):
            bpy.context.user_preferences.system.compute_device_type = 'CUDA'
            bpy.context.user_preferences.system.compute_device = 'CUDA_0'
        else:
            cycles_prefs = bpy.context.user_preferences.addons['cycles'].preferences
            cycles_prefs.compute_device_type = 'CUDA'
        bpy.context.scene.cycles.device = 'GPU'
    render_args = bpy.context.scene.render
    old_filepath = render_args.filepath
    ground = bpy.data.objects['Ground']
    ground.location[0] = ground.location[1] = 0
    ground.rotation_euler[2] = 0
    ground.scale[0] = ground.scale[1] = 100
    ground.scale[2] = 0
    ground_extra = ground.copy()
    ground_extra.name = 'Ground_extra'
    ground_extra.rotation_euler[2] = np.pi
    bpy.context.scene.objects.link(ground_extra)
    delta_theta = (args.max_theta - args.min_theta) / args.num_views
    phi = 0.5 * np.pi * np.random.uniform(args.min_phi, args.max_phi)
    rho = np.random.uniform(args.min_rho, args.max_rho)
    for idx in range(args.num_views):
        # theta = 2 * np.pi * np.random.uniform(args.min_theta, args.max_theta)
        # phi = 0.5 * np.pi * np.random.uniform(args.min_phi, args.max_phi)
        # rho = np.random.uniform(args.min_rho, args.max_rho)
        theta = 2 * np.pi * (args.min_theta + delta_theta * idx)
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
            utils.set_layer(obj, 2)
        #         render_args.filepath = os.path.join(folder, 'back_{}.png'.format(idx))
        #         bpy.ops.render.render(write_still=True)
        # Objects
        for i, obj in enumerate(blender_objects):
            utils.set_layer(obj, 0)
            #             render_args.filepath = os.path.join(folder, 'object_{}_{}.png'.format(idx, i))
            #             bpy.ops.render.render(write_still=True)
            generate_mask(os.path.join(folder, 'mask_{}_{}.png'.format(idx, i)), blender_objects)
            utils.set_layer(obj, 2)
        for i, obj in enumerate(blender_objects):
            utils.set_layer(obj, 0)
    render_args.filepath = old_filepath
    return


def main():
    for idx in range(args.offset, args.offset + args.num_images):
        np.random.seed(idx)
        filename_zip = 'CLEVR_new_{:06d}.blend.gz'.format(idx)
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
        blender_objects = [bpy.data.objects[n['name']] for n in scene['objects']]
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
    argv = utils.extract_args()
    args = parser.parse_args(argv)
    main()
