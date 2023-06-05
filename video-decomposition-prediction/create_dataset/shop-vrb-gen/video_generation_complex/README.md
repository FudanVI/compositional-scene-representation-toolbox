# SHOP-VRB Image Generation


We use Blender to render images for the scenes we generate. In order to invoke `render_images.py` script within Blender engine, call:

```
blender --background --python render_images.py -- [args]
```

where `blender` correspond to the alias of your Blender binary; `--background` and `--python` disable the GUI and call external Python script to be executed (here `render_images.py`); arguments following the `--` will be passed to the `render_images.py`.


## Setup


We use [Blender](https://www.blender.org/) to render synthetic images of the scenes along with JSON file with ground-truth information about the scene.

We suggest using Blender in version at least 2.8x as it comes with denoising filters that allow to render glossy surfaces with much lower cost and without *firefly* artifacts. Additionally, newer Blender version support rendering with OptiX - Nvidia engine for RTX cards (slightly faster than rendering with CUDA).

Blender comes with Python installation built-in. However, we need to add some modules (i.e. *pycocotools*). You can add it directly to Blender Python installation or link other Python installation to Blender (you can symlink your Python to the Blender directory), e.g:

```bash
ln -s /home/uname/anaconda3/envs/blender_python /home/uname/.local/share/blender/2.81/python
```

Remember that when using the latter way, you need to match Python version to the one bundled with your Blender version.

Then, you need to add the `image_generation` directory to Python path of Blender's Python path. You can add a `.pth` file to the `site-packages` directory of Blender's Python, like this:

```bash
echo $PWD/image_generation >> $BLENDER/$VERSION/python/lib/python3.x/site-packages/clevr.pth
```

where `$BLENDER` is the directory where Blender is installed and `$VERSION` is your Blender version.


## Rendering Parameters

### Scene setup

All the scenes are rendered based on the ```--base_scene_blendfile``` (default: ```data/base_scene.blend```). It contains a scene with a ground plane, a camera, and light sources.

SHOP-VRB scenes are generated based on the JSON files constraining the properties of the scenes. Possible colours with RGB values, scaling parameters, material name to blend file mappings, and object models are controlled by ```--properties_json``` (default: ```data/properties.json```). Ground truth information about the objects along with the set of possible modifications (allowed colours, sizes) are set via ```--object_props``` JSON (default: ```data/object_properties.json```). Similarly, ```--grounds``` (default: ```data/ground.json```) defines mapping between ground texture and corresponding files.

The ```.blend``` files containing object models, different materials, and ground textures are stored in ```--shape_dir``` (default: ```data/shapes```), ```--material_dir``` (default: ```data/materials```), ```--ground_dir``` (default: ```data/materials_ground```), respectively.

The ```--boxes_json``` file points to the JSON used to store bounding boxes for all objects used for generating scenes. Boxes are saved in objects' local coordinate systems.

After loading the base scene, the positions of the camera and lights are randomly jittered (controlled with the `--key_light_jitter`, `--fill_light_jitter`, `--back_light_jitter`, and `--camera_jitter` flags).

The set of ground textures can contain non-textured base scene version; controlled by ```--add_plain_ground```.


### Object properties

During generation objects are randomly placed in the scene, one by one. The number of objects in each scene is a random integer between ```--min_objects``` and ```--max_objects```.

In order to assure correct scene generation, the objects are checked whether their bounding boxes are overlapping, and whether they keep minimal distance ```--margin``` in cardinal directions (in order to ensure the clarity of cardinal directions definition - left, right, front, back). Additionally, ```--min_dist``` controls the minimum distance allowed between object centres.

Moreover, we check every scene in terms of object occlusions. We reject the rendering if there exist any object in the scene that has less than ```--min_pixels_per_object``` number of pixels visible. In such way we ensure that there are no fully occluded objects in the scene.

Finally, all the rejection tests are limited to ```--max_retries``` number of tries. After that many unsuccessful attempts of placing each object into the scene, all the objects are re-placed.


### Image Resolution

SHOP-VRB images were rendered at `640x480`, but the resolution can be customized using the `--height` and `--width` flags. After updating the code to use with Blender 2.81, the rendering process is much less time constrained, and the resolution may be increased.


### GPU Acceleration

Rendering uses CPU by default, NVIDIA GPU with CUDA installed can be used for the GPU to accelerate rendering (```--use_gpu```). If you have one of RTX series GPUs you can use new NVIDIA engine - OptiX (```--use_optix```). OptiX is slightly faster than CUDA when rendering SHOP-VRB images.


### Rendering Quality

You can control the quality of rendering with the `--render_num_samples` flag; using fewer samples will run more quickly but will result in grainy images. The number may be smaller when using Blender 2.81 (as it comes with denoiser); for Blender 2.79 and lower you need to use number high enough to get rid of 'fireflies'. The `--render_min_bounces` and `--render_max_bounces` control the number of bounces for transparent objects.

When rendering, Blender breaks up the output image into tiles and renders tiles sequentialy; the `--render_tile_size` flag controls the size of these tiles. This should not affect the output image, but may affect the speed at which it is rendered. For CPU rendering smaller tile sizes may be optimal, while for GPU rendering larger tiles may be faster. You may be able to generate image as one tile on GPU.


### Saving Blender Scene Files

You can save a Blender `.blend` file for each rendered image by adding the flag `--save_blendfiles`. These files can be really big, so they are not saved by default.


### Output Files

Rendered images are stored in the `--output_image_dir` directory, which is created if it does not exist. The filename of each rendered image is constructed from the `--filename_prefix`, the `--split`, and the image index. As you may want to generate images on multiple machines, you may use ```--start_idx``` and ```--num_images``` parameters.

A JSON file for each scene containing ground-truth object positions and attributes is saved in the `--output_scene_dir` directory, which is created if it does not exist. After all images are rendered the JSON files for each individual scene are combined into a single JSON file and written to `--output_scene_file`. This single file will also store the `--split`, `--version`, `--license`, and `--date` (default today).

If saving Blender scene files for each image (`--save_blendfiles`) then they are stored in the `--output_blend_dir` directory, which is created if it does not exist.

