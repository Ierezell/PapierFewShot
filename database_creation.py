import sys
import glob
import argparse
import os

import cv2
import numpy as np
import torch

from face_alignment import FaceAlignment, LandmarksType


face_landmarks = FaceAlignment(LandmarksType._2D, device="cuda")


def parse_args():
    pass


def get_frames(context_path):
    """
    Return the frames of all the videos in the given path

    Arguments:
        context_path {str} -- The path containing the videos

    Returns:
        [list] -- list of the frames of all the videos in context_path
    """

    videos = glob.glob(f"{context_path}/*")
    frames_landmarks = []

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
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                with torch.no_grad():
                    landmark_pts = face_landmarks.get_landmarks_from_image(
                        image)

                try:
                    landmark_pts = landmark_pts[0]
                    landmark_pts[:, 0] = landmark_pts[:, 0] - \
                        min(landmark_pts[:, 0])
                    landmark_pts[:, 1] = landmark_pts[:, 1] - \
                        min(landmark_pts[:, 1])
                    landmark_pts[:, 0] = (
                        landmark_pts[:, 0] / max(landmark_pts[:, 1]))*224
                    landmark_pts[:, 1] = (
                        landmark_pts[:, 1] / max(landmark_pts[:, 1]))*224

                    frames_landmarks.append(landmark_pts)
                except TypeError:
                    continue

        except ValueError:
            continue

    return frames_landmarks


def get_similarity(frames_landmarks):
    """ Create the similarity matrix

    Arguments:
        frames_landmarks {list} -- A list of landmarks

    Returns:
        [numpy.array] -- matrix of the similarity between all the landmarks of 
        frames_landmarks
    """

    N = len(frames_landmarks)

    similarity_matrix = np.zeros(N)

    for i, ldmk in enumerate(frames_landmarks):
        for j in range(i+1, N):
            similarity_matrix[i, j] = np.linalg.norm(
                ldmk - frames_landmarks[j])

    return similarity_matrix


def process(global_path, total_frame_nb):
    context_list = glob.glob(f"{global_path}/*")
    for context in context_list:
        frames_landmarks = get_frames(
            os.path.join(global_path, context))
        similarity_matrix = get_similarity(frames_landmarks)


if __name__ == "__main__":
    args = parse_args()
    process()
