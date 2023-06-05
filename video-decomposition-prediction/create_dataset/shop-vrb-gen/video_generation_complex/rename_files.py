import argparse
import os
import utils


def main():
    prefix = 'SHOP_VRB_train_'
    suffix_scene = '.json'
    suffix_blend = '.blend.gz'
    folders_scene = sorted(os.listdir(args.folder_scene))
    folders_blend = sorted(os.listdir(args.folder_blend))
    assert folders_scene == folders_blend
    for folder in folders_scene:
        filenames_scene = sorted(os.listdir(os.path.join(args.folder_scene, folder)))
        filenames_blend = sorted(os.listdir(os.path.join(args.folder_blend, folder)))
        names_scene, names_blend = [], []
        for filename in filenames_scene:
            assert filename.startswith(prefix)
            assert filename.endswith(suffix_scene)
            names_scene.append(filename[len(prefix):-len(suffix_scene)])
        for filename in filenames_blend:
            assert filename.startswith(prefix)
            assert filename.endswith(suffix_blend)
            names_blend.append(filename[len(prefix):-len(suffix_blend)])
        names_scene = set(names_scene)
        names_blend = set(names_blend)
        names = set.intersection(names_scene, names_blend)
        names_del_scene = names_scene.difference(names)
        names_del_blend = names_blend.difference(names)
        for name in names_del_scene:
            path = os.path.join(args.folder_scene, folder, prefix + name + suffix_scene)
            os.remove(path)
        for name in names_del_blend:
            path = os.path.join(args.folder_blend, folder, prefix + name + suffix_blend)
            os.remove(path)
        names = sorted(list(names))
        for idx, name in enumerate(names):
            path_scene_prev = os.path.join(args.folder_scene, folder, prefix + name + suffix_scene)
            path_scene_new = os.path.join(args.folder_scene, folder, '{}{:06d}{}'.format(prefix, idx, suffix_scene))
            path_blend_prev = os.path.join(args.folder_blend, folder, prefix + name + suffix_blend)
            path_blend_new = os.path.join(args.folder_blend, folder, '{}{:06d}{}'.format(prefix, idx, suffix_blend))
            if path_scene_prev != path_scene_new:
                os.rename(path_scene_prev, path_scene_new)
            if path_blend_prev != path_blend_new:
                os.rename(path_blend_prev, path_blend_new)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder_scene')
    parser.add_argument('--folder_blend')
    argv = utils.extract_args()
    args = parser.parse_args(argv)
    main()
