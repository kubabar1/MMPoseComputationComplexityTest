import time
import datetime
import numpy as np
import cv2
import os
import csv

keypoint_names = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle"
]


def print_progress(frame, start_time, full_frames_count, sequence_nb=None):
    current_time = time.time()
    took_time = current_time - start_time
    if full_frames_count > 0 and frame > 0:
        percent = frame / full_frames_count * 100
        estimated_remaining_time = start_time + (full_frames_count / frame * took_time) - current_time
        x = datetime.timedelta(seconds=int(estimated_remaining_time))
        if sequence_nb is not None:
            print("\r{}: {}% - Estimated remaining time: {}".format(str(sequence_nb), str(round(percent, 2)), str(x)),
                  end='')
        else:
            print("\r{}% - Estimated remaining time: {}".format(str(round(percent, 2)), str(x)), end='')


def print_comparison(predicted, real, keypoint_names, frame_id):
    pos = 0
    key = predicted[pos]['keypoints']
    rel = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    prd = [17, 16, 19, 18, 21, 20, 2, 1, 5, 4, 8, 7]
    print('#######################################################################################')
    for r, p in zip(rel, prd):
        print("%-20s -> pred=(%f, %f)\ttrue=(%f, %f)" % (
            keypoint_names[r], int(key[r][0]), int(key[r][1]), real[0][p][frame_id], real[1][p][frame_id]))
    print('#######################################################################################')


def draw_keypoints_on_video(image, pose_results):
    vis_img_copy = np.copy(image)
    predicted_poses = len(pose_results)
    for pos in range(predicted_poses):
        for i, key in enumerate(pose_results[pos]['keypoints']):
            x_predicted = int(key[0])
            y_predicted = int(key[1])
            cv2.circle(vis_img_copy, (x_predicted, y_predicted), 5, (0, 0, 255), 1)
    return vis_img_copy


def save_keypoints_csv(pose_results, output_csv_root, frame, input_name):
    predicted_poses = len(pose_results)
    for pose_idx in range(predicted_poses):
        keypoint_array_tmp = [frame]
        for keypoint in pose_results[pose_idx]['keypoints']:
            keypoint_array_tmp.extend(keypoint)
        filename = input_name.split(os.path.sep)[-1] + '_pose_' + str(pose_idx) + '.csv'
        output_csv = os.path.join(output_csv_root, filename)
        with open(output_csv, 'a') as myfile:
            wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
            wr.writerow(keypoint_array_tmp)


def save_bounding_box_csv(bboxes, output_csv_root, frame, input_name):
    predicted_poses = len(bboxes)
    for pose_idx in range(predicted_poses):
        bboxes_array_tmp = [frame]
        bboxes_array_tmp.extend(bboxes[pose_idx]['bbox'])
        filename_bbox = input_name.split(os.path.sep)[-1] + '_pose_bounding_box_' + str(pose_idx) + '.csv'
        output_csv_bbox = os.path.join(output_csv_root, filename_bbox)
        with open(output_csv_bbox, 'a') as myfile:
            wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
            wr.writerow(bboxes_array_tmp)


































