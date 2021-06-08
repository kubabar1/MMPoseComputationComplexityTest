from configparser import ConfigParser
import os

from utils.top_down_configs import top_down_configs
from utils.bottom_up_config import bottom_up_configs
from utils.pose_estimation import estimate_pose_movie
from utils.pose_estimation_utils import keypoint_names


def main():
    config = ConfigParser()
    config.read('./config.ini')
    MMPOSE_PATH = config.get('main', 'MMPOSE_PATH')
    test_cases_dataset_path = config.get('main', 'TEST_CASES_DATASET_PATH')
    video_paths = [f.path for f in os.scandir(os.path.join(test_cases_dataset_path)) if f.is_file()]
    output_directory = './results/test_results'

    det_config = os.path.join(MMPOSE_PATH, 'demo/mmdetection_cfg/faster_rcnn_r50_fpn_1x_coco.py')
    det_checkpoint = 'http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'

    repetition_count = 1

    model_type = 'top_down'
    output_analysis_path = os.path.join(output_directory, model_type + '.txt')
    if os.path.exists(output_analysis_path):
        os.remove(output_analysis_path)

    for mode_config in top_down_configs(MMPOSE_PATH):
        pose_config = mode_config[0]
        pose_checkpoint = mode_config[1]
        for movie_nb, video_path in enumerate(video_paths):
            for i in range(repetition_count):
                video_process_time = estimate_pose_movie(det_config, det_checkpoint, pose_config, pose_checkpoint, video_path,
                                                         output_directory,
                                                         keypoint_names, movie_nb, kpt_thr=0.3, bbox_thr=0.3,
                                                         video_dir_name_custom=os.path.basename(video_path).split('.')[0],
                                                         model_type=model_type,
                                                         show=False, save_keypoints=False, save_out_video=False,
                                                         save_bounding_boxes=False)
                print('{}, {}'.format(video_path, video_process_time))
                with open(output_analysis_path, 'a') as f:
                    print('{}, {}'.format(os.path.basename(video_path), video_process_time), file=f)

    model_type = 'bottom_up'
    output_analysis_path = os.path.join(output_directory, model_type + '.txt')
    if os.path.exists(output_analysis_path):
        os.remove(output_analysis_path)

    for mode_config in bottom_up_configs(MMPOSE_PATH):
        pose_config = mode_config[0]
        pose_checkpoint = mode_config[1]
        for movie_nb, video_path in enumerate(video_paths):
            for i in range(repetition_count):
                video_process_time = estimate_pose_movie(det_config, det_checkpoint, pose_config, pose_checkpoint, video_path,
                                                         output_directory,
                                                         keypoint_names, movie_nb, kpt_thr=0.3, bbox_thr=0.3,
                                                         video_dir_name_custom=os.path.basename(video_path).split('.')[0],
                                                         model_type=model_type,
                                                         show=False, save_keypoints=False, save_out_video=False,
                                                         save_bounding_boxes=False)
                print('{}, {}'.format(os.path.basename(video_path), video_process_time))
                with open(output_analysis_path, 'a') as f:
                    print('{}, {}'.format(video_path, video_process_time), file=f)


if __name__ == '__main__':
    main()
