import argparse
import glob
import os
import platform
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


#TODO : frames + ldmk + (frame + ldmk)
# TODO : gérer les / et \\ pour windows
# if Windows in platform.system()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("global_video_path",
                        help="Path to the contexts containing the videos")
    parser.add_argument("global_image_path",
                        help="Path to the contexts containing the images")
    parser.add_argument("total_frame_nb", type=int,
                        help="Number of frames we want to extract per context")
    args = parser.parse_args()

    error_flag = 0

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


def write_landmarks_on_image(image, landmarks):
        # Machoire
    cv2.polylines(image, [np.int32(landmarks[0:17])],
                  isClosed=False, color=(0, 255, 0))
    # Sourcil Gauche
    cv2.polylines(image, [np.int32(landmarks[17:22])],
                  isClosed=False, color=(255, 0, 0))
    # Sourcil droit
    cv2.polylines(image, [np.int32(landmarks[22:27])],
                  isClosed=False, color=(255, 0, 0))
    # Nez arrete
    cv2.polylines(image, [np.int32(landmarks[27:31])],
                  isClosed=False, color=(255, 0, 255))
    # Nez narine
    cv2.polylines(image, [np.int32(landmarks[31:36])],
                  isClosed=False, color=(255, 0, 255))
    # Oeil gauche
    cv2.polylines(image, [np.int32(landmarks[36:42])],
                  isClosed=True, color=(0, 0, 255))
    # oeil droit
    cv2.polylines(image, [np.int32(landmarks[42:48])],
                  isClosed=True, color=(0, 0, 255))
    # Bouche exterieur
    cv2.polylines(image, [np.int32(landmarks[48:60])],
                  isClosed=True, color=(255, 255, 0))
    # Bouche interieur
    cv2.polylines(image, [np.int32(landmarks[60:68])],
                  isClosed=True, color=(255, 255, 0))
    return image


def get_frames(context_path):
    """
    Return the landmarks of all the frames of all the videos in the given path
    and the frames of all the videos

    Arguments:
        context_path {str} -- The path containing the videos

    Returns:
        [list, list] -- list of the frames and of the landmarks
    """

    videos = glob.glob(f"{context_path}/*")
    frames_landmarks = []
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

                    frames.append(image)
                    frames_landmarks.append(landmark_pts)
                except TypeError:
                    continue

        except ValueError:
            continue

    return frames, frames_landmarks


def get_similarity(frames_landmarks):
    """ Create the similarity matrix

    Arguments:
        frames_landmarks {list} -- A list of landmarks
final_path
    Returns:
        [numpy.array] -- matrix of the similarity between all the landmarks of 
        frames_landmarks
    """

    N = len(frames_landmarks)

    similarity_matrix = np.zeros((N, N))

    for i, ldmk_1 in enumerate(frames_landmarks):
        for j, lmdk_2 in enumerate(frames_landmarks[i+1:]):
            similarity_matrix[i, j] = np.linalg.norm(
                ldmk_1 - lmdk_2)

    return similarity_matrix


def select_images(similarity_matrix, total_frame_nb):
    """ Compute the similarity score of each image and returns the ranking

    Arguments:
        similarity_matrix {numpy.array} -- Matrix of the similarity between
        all the frames of the context
        total_frame_nb -- The number of frames we want

    Returns:
        [type] -- [description]
    """

    N = len(similarity_matrix)
    score_of_images = []
    total_matrix = similarity_matrix + similarity_matrix.T

    for i in range(N):
        score_of_images.append((np.linalg.norm(total_matrix[i]), i))

    if N > total_frame_nb:
        score_of_images.sort()
        score_of_images.reverse()
        return score_of_images[:total_frame_nb]

    else:
        return score_of_images


def process(global_video_path, global_image_path, total_frame_nb):

    person_list = glob.glob(f"{global_video_path}/*")
    N = len(person_list)

    for i, person in enumerate(person_list):

        print()
        print(f"Progression : {i+1}/{N}")

        person_name = person.split(slash)[-1]
        context_list = glob.glob(f"{person}/*")

        for j, context in enumerate(context_list):

            context_nb = len(context_list)
            progress(j+1, context_nb, context)

            context_name = context.split(slash)[-1]
            res_path = os.path.join(
                global_image_path, person_name, context_name)

            if not os.path.exists(res_path):
                os.mkdir(res_path)
            frames, frames_landmarks = get_frames(context)
            similarity_matrix = get_similarity(frames_landmarks)
            score_of_images = select_images(similarity_matrix, total_frame_nb)
            for k, score in enumerate(score_of_images):
                black_im = np.zeros(SIZE, np.float32)
                ldmk_im = write_landmarks_on_image(
                    black_im, frames_landmarks[score[1]])
                cplt_im = write_landmarks_on_image(
                    frames[score[1]], frames_landmarks[score[1]])
                cv2.imwrite(os.path.join(
                    res_path, f"frames{k:04d}.jpg"), frames[score[1]])
                cv2.imwrite(os.path.join(
                    res_path, f"frames{k:04d}_ldmk.jpg"), ldmk_im)
                cv2.imwrite(os.path.join(
                    res_path, f"frames{k:04d}_cplt.jpg"), cplt_im)


if __name__ == "__main__":
    args = parse_args()
    process(args.global_video_path, args.global_image_path, args.total_frame_nb)
