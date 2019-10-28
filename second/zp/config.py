from second.zp.utils_zp import *
'''
g_data_root='/home/zp/data/nuscene/nuscene'
g_data_name='nuscene-trainval'
g_version='v1.0-trainval'
'''
g_data_root='/home/zp/data/nuscene/nuscene-mini'
g_data_name='nuscene-mini'
g_version='v1.0-mini'

g_DataPrepare_config=g_data_root + '/DataPrepare.config'
g_VoxelGenerator_config=g_data_root + '/VoxelGenerator.config'

def get_parser_DataPrepare():
    parser = argparse.ArgumentParser(
        description='DataPrepare')
    parser.add_argument(
        '--data_name',default=g_data_name)
    parser.add_argument(
        '--cache_name',default='DataPrepare_cache.pkl')
    parser.add_argument(
        '--data_root',default=g_data_root)
    parser.add_argument(
        '--version',default=g_version)
    parser.add_argument(
        '--phase',default='train')
    parser.add_argument(
        '--verbose',default=True)
    parser.add_argument(
        '--seq_length',default=40)
    parser.add_argument(
        '--obs_length',default=20)
    parser.add_argument(
        '--pred_length',default=20)
    parser.add_argument(
        '--interval',default=1)
    parser.add_argument(
        '--use_image',default='last_image',# 'last_image' or 'key_images'
        help='last_image or key_images')
    return parser

def get_parser_VoxelGenerator():
    parser = argparse.ArgumentParser(
        description='VoxelGenerator')
    parser.add_argument(
        '--full_empty_part_with_mean',default=False)
    parser.add_argument(
        '--point_cloud_range',default=[-72, -40, -2, 72, 40, 5])
    parser.add_argument(
        '--voxel_size',default=[0.2, 0.2, 0.2])
    parser.add_argument(
        '--max_number_of_points_per_voxel',default=40)
    parser.add_argument(
        '--block_filtering',default=False,
        help='filter voxels by block height')
    parser.add_argument(
        '--block_factor',default=1,
        help='height calc width: voxel_size * block_factor * block_size= (0.2 * 1 * 8) ')
    parser.add_argument(
        '--block_size',default=3)
    parser.add_argument(
        '--height_threshold',default=0.2,
        help='locations with height < height_threshold will be removed.')

    parser.add_argument(
        '--bev_data',default=['bev_img','bev_index'])
    return parser

def Configuration(*config_items):
    args={}
    for config_item in config_items:
        get_parser=globals()['get_parser_'+config_item]
        config_path=globals()['g_'+config_item+'_config']

        parser = get_parser()
        p = parser.parse_args()
        if not load_arg(p,config_path):
            save_arg(p,config_path)
        args[config_item]=p
    return args
