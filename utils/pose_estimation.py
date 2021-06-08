import shutil
import time
import timeit

from mmdet.apis import inference_detector, init_detector
from mmpose.apis import (inference_top_down_pose_model, init_pose_model, vis_pose_result, inference_bottom_up_pose_model)
import cv2
import numpy as np
import scipy.io

import os

from utils.pose_estimation_utils import print_progress, draw_keypoints_on_video, save_keypoints_csv, print_comparison, \
    save_bounding_box_csv


def estimate_pose_movie(det_config, det_checkpoint, pose_config, pose_checkpoint, video_path, output_directory,
                        keypoint_names, movie_nb=None, device='cuda:0', return_heatmap=False, kpt_thr=0.3, bbox_thr=0.3,
                        output_layer_names=None, show=True, save_keypoints=True, save_out_video=True,
                        true_coordinates=None, save_bounding_boxes=True, model_type='top_down', video_dir_name_custom=None):
    video_writer = None
    video_dir_name = os.path.basename(os.path.dirname(video_path)) if not video_dir_name_custom else video_dir_name_custom
    model_name = os.path.splitext(os.path.basename(pose_config))[0]
    out_video_root = None
    output_keypoints_root = None
    output_bbox_root = None
    if save_keypoints or save_bounding_boxes or save_out_video:
        out_video_root, output_keypoints_root, output_bbox_root = create_output_directories(output_directory,
                                                                                            model_type, model_name,
                                                                                            video_dir_name,
                                                                                            save_out_video,
                                                                                            save_keypoints,
                                                                                            save_bounding_boxes)

    det_model = init_detector(det_config, det_checkpoint, device=device)
    pose_model = init_pose_model(pose_config, pose_checkpoint, device=device)
    dataset = pose_model.cfg.data['test']['type']

    cap = cv2.VideoCapture(video_path)

    if save_out_video:
        fps = cap.get(cv2.CAP_PROP_FPS)
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(os.path.join(out_video_root, f'vis_{os.path.basename(video_path)}'), fourcc, fps, size)

    frame = 0
    video_process_time = 0
    start_time = time.time()
    full_frames_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    while cap.isOpened():
        flag, img = cap.read()
        if not flag:
            break
        input_name = os.path.splitext(os.path.basename(video_path))[0]
        video_process_time += estimate_pose(img, det_model, pose_model, video_writer, frame, start_time, input_name, dataset,
                                            keypoint_names, output_keypoints_root, output_bbox_root, full_frames_count, bbox_thr,
                                            return_heatmap, output_layer_names, kpt_thr, true_coordinates, show, save_out_video,
                                            save_keypoints, save_bounding_boxes, movie_nb, model_type)
        frame += 1
    cap.release()
    if save_out_video:
        video_writer.release()
    return video_process_time


def process_mmdet_results(mmdet_results, cat_id=0):
    """Process mmdet results, and return a list of bboxes.

    :param mmdet_results:
    :param cat_id: category id (default: 0 for human)
    :return: a list of detected bounding boxes
    """
    if isinstance(mmdet_results, tuple):
        det_results = mmdet_results[0]
    else:
        det_results = mmdet_results

    bboxes = det_results[cat_id]

    person_results = []
    for bbox in bboxes:
        person = {'bbox': bbox}
        person_results.append(person)

    return person_results


def create_output_directories(output_directory, model_type, model_name, video_dir_name, save_out_video, save_keypoints,
                              save_bounding_boxes):
    out_root = os.path.join(output_directory, model_type, model_name)
    out_video_root = os.path.join(out_root, 'videos', video_dir_name)
    output_keypoints_root = os.path.join(out_root, 'csv', video_dir_name, 'pose_results')
    output_bbox_root = os.path.join(out_root, 'csv', video_dir_name, 'bbox')

    video_dir_path = os.path.join(out_root, 'csv', video_dir_name)
    if os.path.exists(video_dir_path) and os.path.isdir(video_dir_path):
        shutil.rmtree(video_dir_path)

    if save_out_video:
        os.makedirs(out_video_root, exist_ok=True)
    if save_keypoints:
        os.makedirs(output_keypoints_root, exist_ok=True)
    if save_bounding_boxes:
        os.makedirs(output_bbox_root, exist_ok=True)
    return out_video_root, output_keypoints_root, output_bbox_root


def estimate_pose(img, det_model, pose_model, video_writer, frame, start_time, input_name, dataset, keypoint_names,
                  output_keypoints_root, output_bbox_root, full_frames_count, bbox_thr, return_heatmap,
                  output_layer_names, kpt_thr, true_coordinates, show, save_out_video, save_keypoints,
                  save_bounding_boxes, movie_nb, model_type):
    if model_type == 'top_down':
        start_inference_time = time.time()
        # test a single image, the resulting box is (x1, y1, x2, y2)
        mmdet_results = inference_detector(det_model, img)

        # keep the person class bounding boxes.
        person_bboxes = process_mmdet_results(mmdet_results)

        # test a single image, with a list of bboxes.
        pose_results, returned_outputs = inference_top_down_pose_model(
            pose_model,
            img,
            person_bboxes,
            bbox_thr=bbox_thr,
            format='xyxy',
            dataset=dataset,
            return_heatmap=return_heatmap,
            outputs=output_layer_names)
        end_inference_time = time.time()
    else:
        start_inference_time = time.time()
        pose_results, returned_outputs = inference_bottom_up_pose_model(
            pose_model,
            img,
            return_heatmap=return_heatmap,
            outputs=output_layer_names)
        end_inference_time = time.time()
    took_time = end_inference_time - start_inference_time

    vis_img = vis_pose_result(
        pose_model,
        img,
        pose_results,
        dataset=dataset,
        kpt_score_thr=kpt_thr,
        show=False)

    if true_coordinates:
        mat = scipy.io.loadmat(true_coordinates)
        print_comparison(pose_results, np.array(mat['joints2D']), keypoint_names, frame)

    if save_out_video:
        video_writer.write(vis_img)

    if save_keypoints:
        save_keypoints_csv(pose_results, output_keypoints_root, frame, input_name)

    if save_bounding_boxes and model_type == 'top_down':
        save_bounding_box_csv(person_bboxes, output_bbox_root, frame, input_name)

    step_time_estimate = 10

    if not (frame % step_time_estimate):
        print_progress(frame, start_time, full_frames_count, movie_nb)

    return took_time
