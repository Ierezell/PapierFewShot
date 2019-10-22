import argparse
import base64
import glob
import json
import os
import platform
import pickle
import sys

import cv2
import numpy as np
import torch
from face_alignment import FaceAlignment, LandmarksType

SIZE = (224, 224)

face_landmarks = FaceAlignment(LandmarksType._2D, device="cuda")
slash = "/"
if "Windows" in platform.system():
    slash = "\\"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("global_video_path",
                        help="Path to the contexts containing the videos")
    parser.add_argument("global_image_path",
                        help="Path to the contexts containing the images")
    parser.add_argument("total_frame_nb", type=int,
                        help="Number of frames we want to extract per context")
    parser.add_argument("from_i", type=int,
                        help="index where we want to start")
    parser.add_argument("to_i", type=int, help="index where we want to stop")

    args = parser.parse_args()

    error_flag = 0
    if args.to_i != -1:
        if args.from_i > args.to_i:
            print("The value of from must be smaller than the value of to")
            error_flag = 1

    if not os.path.exists(args.global_video_path):
        print("The path " + args.global_video_path + " does not exist")
        error_flag = 1

    if not os.path.exists(args.global_image_path):
        print("The path " + args.global_image_path + " does not exist")
        error_flag = 1

    if error_flag:
        sys.exit(1)

    return args


def progress(count, total, in_progress=""):
    """
    Progress of the algorithm : display the percentage of the progress,
    and the file which is in treatment
    """

    percents = count / total * 100

    sys.stdout.write(f"{percents:.1f}% Context : {in_progress}\r")
    sys.stdout.flush()


def get_frames(context_path):
    """
    Return all the frames of the videos of context_path

    Arguments:
        context_path {str} -- The path containing the videos

    Returns:
        [list] -- list of the frames
    """

    videos = glob.glob(f"{context_path}/*.mp4")
    frames = []

    for v in videos:

        try:
            i = 0
            video = cv2.VideoCapture(v)

            while(video.isOpened()):
                ret, image = video.read()
                if ret == False:
                    break

                image = cv2.flip(image, 1)
                image = cv2.resize(image, (224, 224))

                frames.append(image)

        except ValueError:
            continue

    return frames


def select_images(frames, total_frame_nb):
    """ Compute the similarity score of each image and returns the ranking

    Arguments:
        frames {list of np.array} -- list of frames
        total_frame_nb -- The number of frames we want

    Returns:
        [list] -- a list of total_frame_nb frames
    """

    N = len(frames)
    res = []

    if N > total_frame_nb:
        choice = np.random.choice(N, total_frame_nb, replace=False)
        for i in choice:
            res.append(frames[i])
        return res

    else:
        return frames


def get_ldmk(frames):
    """ Gete the landmarks of a list of frames

    Arguments:
        frames {list of np.array} -- list of frames

    Returns:
        [list] -- a list of landmarks
    """

    frames_landmarks = []

    for f in frames:
        with torch.no_grad():
            landmark_pts = face_landmarks.get_landmarks_from_image(f)

        try:
            landmark_pts = landmark_pts[0]
            frames_landmarks.append(landmark_pts)
        except TypeError:
            frames.append(np.array([]))

    return frames_landmarks


def process(global_video_path, global_image_path, total_frame_nb, from_i, to_i):

    person_list = glob.glob(f"{global_video_path}/*")
    if to_i != -1:
        N = to_i - from_i
    else:
        N = len(person_list) - from_i

    for i, person in enumerate(person_list[from_i:to_i]):

        print()
        print(f"Progression : {i+1}/{N}")

        person_name = person.split(slash)[-1]
        context_list = glob.glob(f"{person}/*")

        person_path = os.path.join(global_image_path, person_name)
        if not os.path.exists(person_path):
            os.mkdir(person_path)

        for j, context in enumerate(context_list):

            context_nb = len(context_list)
            progress(j+1, context_nb, context)

            context_name = context.split(slash)[-1]
            # res_path = os.path.join(
            #     global_image_path, person_name, context_name)
            res_path = os.path.join(person_path, context_name)

            if not os.path.exists(res_path):
                os.mkdir(res_path)

            frames = get_frames(context)
            images_list = select_images(frames, total_frame_nb)
            images_ldmk = get_ldmk(images_list)

            data = {}
            data['frames'] = []

            for k in range(len(images_list)):

                imdata = pickle.dumps(images_list[k])

                data['frames'].append({
                    # 'frame': base64.b64encode(imdata).decode('ascii'),
                    'frame': k,
                    'ldmk': images_ldmk[k].tolist()
                })

            with open(os.path.join(res_path, "frames.json"), 'w') as file:
                json.dump(data, file)


if __name__ == "__main__":
    args = parse_args()
    process(args.global_video_path, args.global_image_path,
            args.total_frame_nb, args.from_i, args.to_i)
