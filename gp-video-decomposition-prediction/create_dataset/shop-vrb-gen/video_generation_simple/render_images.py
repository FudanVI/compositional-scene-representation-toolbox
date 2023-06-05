'''
Based on:
clevr-dataset-gen
https://github.com/facebookresearch/clevr-dataset-gen
'''

from __future__ import print_function

import pdb
import sys
import random
import argparse
import json
import os
import tempfile
from datetime import datetime as dt
from collections import Counter
import numpy as np
import gzip, shutil

"""
Renders random scenes using Blender, each with with a random number of objects;
each object has a random size, position, color, and shape. Objects will be
nonintersecting but may partially occlude each other. Output images will be
written to disk as PNGs, and we will also write a JSON file for each image with
ground-truth scene information, such as given properties and encoded segmentation mask.

This file expects to be run from Blender like this:

blender --background --python render_images.py -- [arguments to this script]
"""

INSIDE_BLENDER = True
try:
    import bpy
    from mathutils import Vector
except ImportError:
    INSIDE_BLENDER = False
if INSIDE_BLENDER:
    try:
        import utils
    except ImportError:
        print("\nERROR")
        print("Running render_images.py from Blender and cannot import utils.py.")
        print("You may need to add a .pth file to the site-packages of Blender's")
        print("bundled python with a command like this:\n")
        print("echo $PWD >> $BLENDER/$VERSION/python/lib/python3.7/site-packages/shop_vrb.pth")
        print("\nWhere $BLENDER is the directory where Blender is installed, and")
        print("$VERSION is your Blender version (such as 2.81).")
        sys.exit(1)

parser = argparse.ArgumentParser()

# Customized options
parser.add_argument('--seed', type=int, default=None)
parser.add_argument('--crop_up', type=int, default=None)
parser.add_argument('--crop_down', type=int, default=None)
parser.add_argument('--crop_left', type=int, default=None)
parser.add_argument('--crop_right', type=int, default=None)

# Input options
parser.add_argument(
    '--base_scene_blendfile',
    default='../image_generation/data/base_scene.blend',
    help="Base blender file on which all scenes are based; includes " +
    "ground plane, lights, and camera.")
parser.add_argument(
    '--properties_json', default='data/properties.json',
    help="JSON file defining objects, materials, sizes, and colors. " +
    "The \"colors\" field maps from CLEVR color names to RGB values; " +
    "The \"sizes\" field maps from CLEVR size names to scalars used to " +
    "rescale object models; the \"materials\" and \"shapes\" fields map " +
    "from CLEVR material and shape names to .blend files in the " +
    "--object_material_dir and --shape_dir directories respectively.")
parser.add_argument(
    '--object_props',
    default='data/object_properties.json',
    help="JSON file defining properties of objects. " +
    "Properties are ground truth information about the objects along with indication " +
    "whether material, colour or size should be changed.")
parser.add_argument(
    '--grounds',
    default='data/ground.json',
    help="JSON file defining ground textures, pointing to .blend files")
parser.add_argument(
    '--shape_dir',
    default='../image_generation/data/shapes',
    help="Directory where .blend files for object models are stored")
parser.add_argument(
    '--material_dir',
    default='../image_generation/data/materials',
    help="Directory where .blend files for materials are stored")
parser.add_argument(
    '--ground_dir',
    default='../image_generation/data/materials_ground',
    help="Directory where .blend files for ground materials are stored")

# Settings for objects
parser.add_argument(
    '--min_objects', default=3, type=int,
    help="The minimum number of objects to place in each scene")
parser.add_argument(
    '--max_objects', default=10, type=int,
    help="The maximum number of objects to place in each scene")
parser.add_argument(
    '--min_dist', default=0.25, type=float,
    help="The minimum allowed distance between object centers")
parser.add_argument(
    '--margin', default=0.2, type=float,
    help="Along all cardinal directions (left, right, front, back), all " +
    "objects will be at least this distance apart. This makes resolving " +
    "spatial relationships slightly less ambiguous.")
parser.add_argument(
    '--min_pixels_per_object', default=200, type=int,
    help="All objects will have at least this many visible pixels in the " +
    "final rendered images; this ensures that no objects are fully " +
    "occluded by other objects.")
parser.add_argument(
    '--max_retries', default=50, type=int,
    help="The number of times to try placing an object before giving up and " +
    "re-placing all objects in the scene.")
parser.add_argument(
    '--add_plain_ground', default=False, action='store_true',
    help="Add plain ground (no texture) to the set of possibilities.")

# Output settings
parser.add_argument(
    '--start_idx', default=0, type=int,
    help="The index at which to start for numbering rendered images. Setting " +
    "this to non-zero values allows you to distribute rendering across " +
    "multiple machines and recombine the results later.")
parser.add_argument(
    '--num_images', default=5, type=int,
    help="The number of images to render")
parser.add_argument(
    '--filename_prefix', default='SHOP_VRB',
    help="This prefix will be prepended to the rendered images and JSON scenes")
parser.add_argument(
    '--split', default='train',
    help="Name of the split for which we are rendering. This will be added to " +
    "the names of rendered images, and will also be stored in the JSON " +
    "scene structure for each image.")
parser.add_argument(
    '--output_image_dir', default='../output/images/',
    help="The directory where output images will be stored. It will be " +
    "created if it does not exist.")
parser.add_argument(
    '--output_scene_dir', default='../output/scenes/',
    help="The directory where output JSON scene structures will be stored. " +
    "It will be created if it does not exist.")
parser.add_argument(
    '--output_blend_dir', default='output/blendfiles',
    help="The directory where blender scene files will be stored, if the " +
    "user requested that these files be saved using the " +
    "--save_blendfiles flag; in this case it will be created if it does " +
    "not already exist.")
parser.add_argument(
    '--save_blendfiles', default=False, action='store_true',
    help="Setting --save_blendfiles will cause the blender scene file for " +
    "each generated image to be stored in the directory specified by " +
    "the --output_blend_dir flag. These files are not saved by default " +
    "because they take up a lot of space.")
parser.add_argument(
    '--version', default='1.0',
    help="String to store in the \"version\" field of the generated JSON file")
parser.add_argument(
    '--license', default="Creative Commons Attribution (CC-BY 4.0)",
    help="String to store in the \"license\" field of the generated JSON file")
parser.add_argument(
    '--date', default=dt.today().strftime("%d/%m/%Y"),
    help="String to store in the \"date\" field of the generated JSON file; " +
    "defaults to today's date")

# Rendering options
parser.add_argument(
    '--use_gpu', default=False, action='store_true',
    help="Setting --use_gpu enables GPU-accelerated rendering using CUDA. " +
    "You must have an NVIDIA GPU with the CUDA toolkit installed for " +
    "to work.")
parser.add_argument(
    '--use_optix', default=False, action='store_true',
    help="Setting --use_optix enables GPU-accelerated rendering using OptiX. " +
    "You must have an NVIDIA RTX with Nvidia drivers >=435 " +
    "to work. Faster than CUDA. Must set use_gpu before.")
parser.add_argument(
    '--width', default=640, type=int,
    help="The width (in pixels) for the rendered images")
parser.add_argument(
    '--height', default=480, type=int,
    help="The height (in pixels) for the rendered images")
parser.add_argument(
    '--key_light_jitter', default=1.0, type=float,
    help="The magnitude of random jitter to add to the key light position.")
parser.add_argument(
    '--fill_light_jitter', default=1.0, type=float,
    help="The magnitude of random jitter to add to the fill light position.")
parser.add_argument(
    '--back_light_jitter', default=1.0, type=float,
    help="The magnitude of random jitter to add to the back light position.")
parser.add_argument(
    '--camera_jitter', default=0.5, type=float,
    help="The magnitude of random jitter to add to the camera position")
parser.add_argument(
    '--render_num_samples', default=64, type=int,
    help="The number of samples to use when rendering. Larger values will " +
    "result in nicer images but will cause rendering to take longer. " +
    "No need to be that high since Blender 2.8 which has built-in denoiser.")
parser.add_argument(
    '--render_min_bounces', default=8, type=int,
    help="The minimum number of bounces to use for rendering.")
parser.add_argument(
    '--render_max_bounces', default=8, type=int,
    help="The maximum number of bounces to use for rendering.")
parser.add_argument(
    '--render_tile_size', default=2048, type=int,
    help="The tile size to use for rendering. This should not affect the " +
    "quality of the rendered image but may affect the speed; CPU-based " +
    "rendering may achieve better performance using smaller tile sizes " +
    "while larger tile sizes may be optimal for GPU-based rendering. " +
    "If it works with your GPU you can set it even to size covering the whole image.")


def main(args):
    num_digits = 6
    prefix = '%s_%s_' % (args.filename_prefix, args.split)
    img_template = '%s%%0%dd.png' % (prefix, num_digits)
    scene_template = '%s%%0%dd.json' % (prefix, num_digits)
    blend_template = '%s%%0%dd.blend' % (prefix, num_digits)
    img_template = os.path.join(args.output_image_dir, img_template)
    scene_template = os.path.join(args.output_scene_dir, scene_template)
    blend_template = os.path.join(args.output_blend_dir, blend_template)

    if not os.path.isdir(args.output_image_dir):
        os.makedirs(args.output_image_dir)
    if not os.path.isdir(args.output_scene_dir):
        os.makedirs(args.output_scene_dir)
    if args.save_blendfiles and not os.path.isdir(args.output_blend_dir):
        os.makedirs(args.output_blend_dir)

    boxes = utils.get_boundings_size(args.properties_json, args.shape_dir)

    all_scene_paths = []
    for i in range(args.num_images):
        img_path = img_template % (i + args.start_idx)
        scene_path = scene_template % (i + args.start_idx)
        all_scene_paths.append(scene_path)
        blend_path = None
        if args.save_blendfiles:
            blend_path = blend_template % (i + args.start_idx)
        num_objects = random.randint(args.min_objects, args.max_objects)
        render_scene(
            args,
            num_objects=num_objects,
            output_index=(i + args.start_idx),
            output_split=args.split,
            output_image=img_path,
            output_scene=scene_path,
            output_blendfile=blend_path,
            boxes=boxes,
        )
    return


def render_scene(
        args,
        num_objects=5,
        output_index=0,
        output_split='none',
        output_image='render.png',
        output_scene='render_json',
        output_blendfile=None,
        boxes=None):

    # Load the main blendfile
    bpy.ops.wm.open_mainfile(filepath=args.base_scene_blendfile)

    # Load materials
    utils.load_materials(args.material_dir)
    utils.load_materials(args.ground_dir)

    with open(args.grounds, 'r') as f:
        grounds = json.load(f)

    grounds = list(grounds.keys())
    if args.add_plain_ground:
        grounds.append(None)

    # Set render arguments so we can get pixel coordinates later.
    # We use functionality specific to the CYCLES renderer so BLENDER_RENDER
    # cannot be used.
    render_args = bpy.context.scene.render
    render_args.engine = "CYCLES"
    render_args.filepath = output_image
    render_args.resolution_x = args.width
    render_args.resolution_y = args.height
    render_args.resolution_percentage = 100
    render_args.tile_x = args.render_tile_size
    render_args.tile_y = args.render_tile_size
    if args.use_gpu:
        # Blender changed the API for enabling CUDA at some point
        if bpy.app.version < (2, 78, 0):
            bpy.context.user_preferences.system.compute_device_type = 'CUDA'
            bpy.context.user_preferences.system.compute_device = 'CUDA_0'
        elif bpy.app.version < (2, 80, 0):
            cycles_prefs = bpy.context.user_preferences.addons['cycles'].preferences
            cycles_prefs.compute_device_type = 'CUDA'
        else:
            cycles_prefs = bpy.context.preferences.addons['cycles'].preferences
            if args.use_optix:
                cycles_prefs.compute_device_type = 'OPTIX'
            else:
                cycles_prefs.compute_device_type = 'CUDA'

    if not (args.use_gpu and args.use_optix):
        bpy.data.worlds['World'].cycles.sample_as_light = True

    # Some CYCLES-specific stuff
    bpy.context.scene.cycles.blur_glossy = 2.0
    bpy.context.scene.cycles.samples = args.render_num_samples
    bpy.context.scene.cycles.transparent_min_bounces = args.render_min_bounces
    bpy.context.scene.cycles.transparent_max_bounces = args.render_max_bounces
    if args.use_gpu:
        bpy.context.scene.cycles.device = 'GPU'
    bpy.context.preferences.addons['cycles'].preferences.get_devices()

    # This will give ground-truth information about the scene and its objects
    scene_struct = {
        'split': output_split,
        'image_index': output_index,
        'image_filename': os.path.basename(output_image),
        'objects': [],
        'directions': {},
    }

    # Put a plane on the ground so we can compute cardinal directions
    bpy.ops.mesh.primitive_plane_add(size=5)
    plane = bpy.context.object

    def rand(L):
        return 2.0 * L * (random.random() - 0.5)

    # Add random jitter to camera position
    if args.camera_jitter > 0:
        for i in range(3):
            bpy.data.objects['Camera'].location[i] += rand(args.camera_jitter)

    # Figure out the left, up, and behind directions along the plane and record
    # them in the scene structure
    camera = bpy.data.objects['Camera']
    plane_normal = plane.data.vertices[0].normal
    cam_behind = camera.matrix_world.to_quaternion() @ Vector((0, 0, -1))
    cam_left = camera.matrix_world.to_quaternion() @ Vector((-1, 0, 0))
    cam_up = camera.matrix_world.to_quaternion() @ Vector((0, 1, 0))
    plane_behind = (cam_behind - cam_behind.project(plane_normal)).normalized()
    plane_left = (cam_left - cam_left.project(plane_normal)).normalized()
    plane_up = cam_up.project(plane_normal).normalized()

    # Delete the plane; we only used it for normals anyway. The base scene file
    # contains the actual ground plane.
    utils.delete_object(plane)

    # Save all six axis-aligned directions in the scene struct
    scene_struct['directions']['behind'] = tuple(plane_behind)
    scene_struct['directions']['front'] = tuple(-plane_behind)
    scene_struct['directions']['left'] = tuple(plane_left)
    scene_struct['directions']['right'] = tuple(-plane_left)
    scene_struct['directions']['above'] = tuple(plane_up)
    scene_struct['directions']['below'] = tuple(-plane_up)

    material = random.choice(grounds)
    if material is not None:
        utils.add_ground(material)

    # Add random jitter to lamp positions
    if args.key_light_jitter > 0:
        for i in range(3):
            bpy.data.objects['Lamp_Key'].location[i] += rand(args.key_light_jitter)
    if args.back_light_jitter > 0:
        for i in range(3):
            bpy.data.objects['Lamp_Back'].location[i] += rand(args.back_light_jitter)
    if args.fill_light_jitter > 0:
        for i in range(3):
            bpy.data.objects['Lamp_Fill'].location[i] += rand(args.fill_light_jitter)

    # Now make some random objects
    if os.path.exists(output_blendfile + '.gz'):
        exit()
    if os.path.exists(output_blendfile):
        os.remove(output_blendfile)
    bpy.ops.wm.save_as_mainfile(filepath=output_blendfile)
    objects, blender_objects = None, None
    num_tries = 0
    while objects is None:
        num_tries += 1
        if num_tries > 1000:
            os.remove(output_blendfile)
            exit()
        bpy.ops.wm.quit_blender()
        bpy.ops.wm.open_mainfile(filepath=output_blendfile)
        utils.load_materials(args.material_dir)
        utils.load_materials(args.ground_dir)
        camera = bpy.data.objects['Camera']
        objects, blender_objects = add_random_objects(scene_struct, num_objects, args, camera, boxes)
        print("\n\nRelocating all objects\n\n")
    os.remove(output_blendfile)

    # Render the scene and dump the scene data structure
    scene_struct['objects'] = objects
    scene_struct['relationships'] = compute_all_relationships(scene_struct)
    while True:
        try:
            if not args.save_blendfiles:
                bpy.ops.render.render(write_still=True)
            break
        except Exception as e:
            print(e)

    with open(output_scene, 'w') as f:
        json.dump(scene_struct, f, indent=2)

    if args.save_blendfiles:
        bpy.ops.wm.save_as_mainfile(filepath=output_blendfile)
        with open(output_blendfile, 'rb') as f_in, gzip.open(output_blendfile + '.gz', 'wb', compresslevel=1) as f_out:
            shutil.copyfileobj(f_in, f_out)
        os.remove(output_blendfile)
    return


def add_random_objects(scene_struct, num_objects, args, camera, boxes):
    """
    Add random objects to the current blender scene
    """

    # Load the property file
    with open(args.properties_json, 'r') as f:
        properties = json.load(f)
        color_name_to_rgba = {}
        for name, rgb in properties['colors'].items():
            rgba = [float(c) / 255.0 for c in rgb] + [1.0]
            color_name_to_rgba[name] = rgba
        object_mapping = [(v, k) for k, v in properties['shapes'].items()]
        size_mapping = properties['sizes']

    white = [0.88, 0.88, 0.88, 1.0]
    pdb.set_trace()
    with open(args.object_props, 'r') as f:
        object_properties = json.load(f)

    positions = []
    objects = []
    blender_objects = []
    sizes = []
    for i in range(num_objects):
        # Choose random color and shape
        obj_name, obj_name_out = random.choice(object_mapping)
        color_name, rgba = random.choice(list(color_name_to_rgba.items()))

        # Choose a random size
        if object_properties[obj_name]['change_size']:
            size_name = random.choice(list(size_mapping.keys()))
            r = size_mapping[size_name]
            if size_name == "bigger":
                size_name = object_properties[obj_name]['size1']
            else:
                size_name = object_properties[obj_name]['size2']
        else:
            size_name = object_properties[obj_name]['size1']
            r = size_mapping["bigger"]

        num_tries = 0
        while True:
            # If we try and fail to place an object too many times, then delete all
            # the objects in the scene and start over.
            num_tries += 1
            if num_tries > args.max_retries:
                for obj in blender_objects:
                    utils.delete_object(obj)
                del blender_objects
                del objects
                # return add_random_objects(scene_struct, num_objects, args, camera)
                return None, None
            x = random.uniform(-3.5, 3.5)
            y = random.uniform(-3.5, 3.5)

            # Choose random orientation for the object.
            theta = 180.0 * random.random() - 120.0

            # Check to make sure the new object is further than min_dist from all
            # other objects, and further than margin along the four cardinal directions
            margins_good = True

            pos_temp = positions.copy()
            pos_temp.append((x, y, r, theta))
            size_temp = sizes.copy()
            size_temp.append(boxes[obj_name])

            for (xx, yy, rr, th) in positions:
                dx, dy = x - xx, y - yy
                for direction_name in ['left', 'right', 'front', 'behind']:
                    direction_vec = scene_struct['directions'][direction_name]
                    assert direction_vec[2] == 0
                    margin = dx * direction_vec[0] + dy * direction_vec[1]
                    if 0 < margin < args.margin:
                        print(margin, args.margin, direction_name)
                        print('Broken margin!')
                        margins_good = False
                        break
                if not margins_good:
                    break

            dists_good = not utils.check_intersection_list(pos_temp, size_temp, args.min_dist)
            del pos_temp

            if dists_good and margins_good:
                break

        # Actually add the object to the scene
        utils.add_object(args.shape_dir, obj_name, r, (x, y), theta=theta)
        obj = bpy.context.object
        blender_objects.append(obj)
        positions.append((x, y, r, theta))
        sizes.append(boxes[obj_name])

        # Attach a random material
        if object_properties[obj_name]['change_material']:
            change = random.choice([True, False])
            if change:
                mat_name_out = object_properties[obj_name]['material2']
                mat_name = mat_name_out.capitalize()
                if object_properties[obj_name]['change_color2']:
                    utils.add_material(mat_name, Color=rgba)
                else:
                    color_name = object_properties[obj_name]['color2']
                    utils.add_material(mat_name, Color=white)
            else:
                mat_name_out = object_properties[obj_name]['material1']
                if object_properties[obj_name]['change_color1']:
                    utils.add_color(rgba)
                else:
                    color_name = object_properties[obj_name]['color1']
        else:
            mat_name_out = object_properties[obj_name]['material1']
            if object_properties[obj_name]['change_color1']:
                utils.add_color(rgba)
            else:
                color_name = object_properties[obj_name]['color1']

        # Record data about the object in the scene data structure
        pixel_coords = utils.get_camera_coords(camera, obj.location)

        obj_name_out = object_properties[obj_name]['name']

        weight = object_properties[obj_name]['weight']
        movability = object_properties[obj_name]['movability']
        shape = object_properties[obj_name]['shape']

        objects.append({
            'name_raw': obj.name,
            'name': obj_name_out,
            'shape': shape,
            'size': size_name,
            'material': mat_name_out,
            '3d_coords': tuple(obj.location),
            'rotation': theta,
            'pixel_coords': pixel_coords,
            'color': color_name,
            'weight': weight,
            'movability': movability
        })

#     # Check that all objects are at least partially visible in the rendered image
#     all_visible = check_visibility(blender_objects, args.min_pixels_per_object, args.height, args.width,
#                                    args.crop_up, args.crop_down, args.crop_left, args.crop_right)
#     if not all_visible:
#         # If any of the objects are fully occluded then start over; delete all
#         # objects from the scene and place them all again.
#         print('Some objects are occluded; replacing objects')
#         for obj in blender_objects:
#             utils.delete_object(obj)
#         del blender_objects
#         del objects
#         return None, None

    return objects, blender_objects


def compute_all_relationships(scene_struct, eps=0.2):
    """
    Computes relationships between all pairs of objects in the scene.

    Returns a dictionary mapping string relationship names to lists of lists of
    integers, where output[rel][i] gives a list of object indices that have the
    relationship rel with object i. For example if j is in output['left'][i] then
    object j is left of object i.
    """
    all_relationships = {}
    for name, direction_vec in scene_struct['directions'].items():
        if name == 'above' or name == 'below':
            continue
        all_relationships[name] = []
        for i, obj1 in enumerate(scene_struct['objects']):
            coords1 = obj1['3d_coords']
            related = set()
            for j, obj2 in enumerate(scene_struct['objects']):
                if obj1 == obj2:
                    continue
                coords2 = obj2['3d_coords']
                diff = [coords2[k] - coords1[k] for k in [0, 1, 2]]
                dot = sum(diff[k] * direction_vec[k] for k in [0, 1, 2])
                if dot > eps:
                    related.add(j)
            all_relationships[name].append(sorted(list(related)))
    return all_relationships


def check_visibility(blender_objects, min_pixels_per_object, height, width, crop_up, crop_down, crop_left, crop_right):
    """
    Check whether all objects in the scene have some minimum number of visible
    pixels; to accomplish this we assign random (but distinct) colors to all
    objects, and render using no lighting or shading or antialiasing; this
    ensures that each object is just a solid uniform color. We can then count
    the number of pixels of each color in the output image to check the visibility
    of each object.

    Returns True if all objects are visible and False otherwise.
    If True returns segmentation mask as well
    """
    f, path = tempfile.mkstemp(suffix='.png')
    # Render shadeless and return list of colours
    object_colors = render_shadeless(blender_objects, path)
    img = bpy.data.images.load(path)
    p = list(img.pixels)
    # Count whether the number of colours is correct - full occlusion
    crop_up = 0 if crop_up is None else crop_up
    crop_down = height if crop_down is None else crop_down
    crop_left = 0 if crop_left is None else crop_left
    crop_right = width if crop_right is None else crop_right
    pixels_flat = [(p[i], p[i + 1], p[i + 2], p[i + 3]) for i in range(0, len(p), 4)]
    pixels_mat = [pixels_flat[i:i + width] for i in range(0, len(pixels_flat), width)]
    pixels_mat_crop = [n[crop_left:crop_right] for n in pixels_mat[crop_up:crop_down]]
    pixels_flat_crop = [m for n in pixels_mat_crop for m in n]
    color_count = Counter(n for n in pixels_flat_crop)
    os.remove(path)
    if len(color_count) != len(blender_objects) + 1:
        return False
    # Check partial occlusion
    for _, count in color_count.most_common():
        if count < min_pixels_per_object:
            return False
    return True


def render_shadeless(blender_objects, path='flat.png'):
    """
    Render a version of the scene with shading disabled and unique materials
    assigned to all objects, and return a set of all colors that should be in the
    rendered image. The image itself is written to path. This is used to ensure
    that all objects will be visible in the final rendered scene.
    """
    render_args = bpy.context.scene.render

    # Cache the render args we are about to clobber
    old_filepath = render_args.filepath
    old_engine = render_args.engine
    old_filter_size = render_args.filter_size

    # Override some render settings to have flat shading
    render_args.filepath = path
    render_args.engine = 'BLENDER_WORKBENCH'
    render_args.filter_size = 0.0

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
    object_colors = []
    new_obj = []
    for obj in bpy.data.objects:
        obj.select_set(state=False)
    for i, obj in enumerate(blender_objects):
        obj.select_set(state=True)
        bpy.ops.object.duplicate(linked=False, mode='INIT')
        utils.set_render(obj, False)
        bpy.ops.material.new()
        mat = bpy.data.materials['Material']
        mat.name = 'Material_temp_%d' % i
        while True:
            r, g, b = [random.randint(0, 255) for _ in range(3)]
            if (r, g, b) not in object_colors and (r, g, b) != (13, 13, 13):
                break
        object_colors.append((r, g, b))
        mat.diffuse_color = (float(r) / 255, float(g) / 255, float(b) / 255, 1.0)
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

    for obj in blender_objects:
        utils.set_render(obj, True)

    # Render lights again
    utils.set_render(bpy.data.objects['Lamp_Key'], True)
    utils.set_render(bpy.data.objects['Lamp_Fill'], True)
    utils.set_render(bpy.data.objects['Lamp_Back'], True)
    utils.set_render(bpy.data.objects['Ground'], True)

    # Set the render settings back to what they were
    render_args.filepath = old_filepath
    render_args.engine = old_engine
    render_args.filter_size = old_filter_size
    bpy.context.scene.display.shading.light = old_shading
    bpy.context.scene.display.render_aa = old_aa
    bpy.context.scene.node_tree.nodes["Switch"].check = old_denoising_state

    return object_colors


if __name__ == '__main__':
    if INSIDE_BLENDER:
        # Run normally
        argv = utils.extract_args()
        args = parser.parse_args(argv)
        if args.seed is not None:
            random.seed(args.seed)
            np.random.seed(args.seed)
        main(args)
    elif '--help' in sys.argv or '-h' in sys.argv:
        parser.print_help()
    else:
        print('This script is intended to be called from blender like this:')
        print()
        print('blender --background --python render_images.py -- [args]')
        print()
        print('You can also run as a standalone python script to view all')
        print('arguments like this:')
        print()
        print('python render_images.py --help')
