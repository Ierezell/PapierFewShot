import argparse
import glob
import json
import os
import platform
import sys
import time
import cv2
import numpy as np
import torch
from face_alignment import FaceAlignment, LandmarksType
from moviepy.editor import VideoFileClip, concatenate_videoclips


torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

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


def get_frames(context_video, nb_frame_to_keep, dict_ldmk):
    """
    Return all the frames of the videos of context_path

    Arguments:
        context_path {str} -- The path containing the videos

    Returns:
        [list] -- list of the frames
    """
    frames = []
    cvVideo = cv2.VideoCapture(context_video)
    total_frame_nb = int(cvVideo.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_already_made = set(int(k) for k, v in dict_ldmk.items())
    frames_to_do = list(set(range(total_frame_nb))-frames_already_made)
    if nb_frame_to_keep < total_frame_nb:
        frameIndexChoice = np.random.choice(frames_to_do, nb_frame_to_keep,
                                            replace=False)
    else:
        frameIndexChoice = frames_to_do

    for frameIndex in set(frameIndexChoice):
        cvVideo.set(cv2.CAP_PROP_POS_FRAMES, frameIndex)
        _, frame = cvVideo.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append((int(frameIndex), frame))
    return frames


def concat_videos(context_path):
    videos = glob.glob(f"{context_path}/*.mp4")
    video_clips = []
    for v in videos:
        video_clips.append(VideoFileClip(v, audio=False))
    final_clip = concatenate_videoclips(video_clips)
    # for v in video_clips:
    #     v.close()
    return final_clip


def get_ldmk(frames):
    """ Gete the landmarks of a list of frames

    Arguments:
        frames {list of np.array} -- list of frames

    Returns:
        [list] -- a list of landmarks
    """

    frames_landmarks = []
    for index, f in frames:
        with torch.no_grad():
            # start = time.time()
            landmark_pts = face_landmarks.get_landmarks_from_image(f)
            # print(f"Time Ldmk : {time.time()-start}s")
        try:
            landmark_pts = landmark_pts[0]
            frames_landmarks.append((index, landmark_pts))
        except TypeError:
            frames_landmarks.append((index, np.array([])))
    return frames_landmarks


def process(global_video_path, global_image_path, nb_frame_to_keep, from_i, to_i):

    person_list = glob.glob(f"{global_video_path}/*")

    if to_i != -1:
        nb_persons = to_i - from_i
    else:
        nb_persons = len(person_list) - from_i

    for i, person in enumerate(person_list[from_i:to_i]):

        print()
        print(f"Progression : {i+1}/{nb_persons}")

        person_name = person.split(slash)[-1]
        context_list = glob.glob(f"{person}/*")

        person_path = os.path.join(global_image_path, person_name)

        if not os.path.exists(person_path):
            os.mkdir(person_path)

        for j, context in enumerate(context_list):
            context_nb = len(context_list)
            # progress(j+1, context_nb, context)

            context_name = context.split(slash)[-1]

            context_video_path = os.path.join(
                person_path, f"{context_name}.mp4")
            if not os.path.exists(context_video_path):
                cat_video = concat_videos(context)
                try:
                    cat_video.write_videofile(context_video_path,
                                              codec="libx264", audio=False,
                                              threads=6, logger=None)
                except IndexError:
                    last_frame = cat_video.duration - 1.0 / cat_video.fps
                    cat_video = cat_video.subclip(t_end=(last_frame))

                    cat_video.write_videofile(context_video_path,
                                              codec="libx264",
                                              audio=False,
                                              threads=6, logger=None)

            json_path = os.path.join(person_path, f"{context_name}.json")
            if not os.path.exists(json_path):
                dict_ldmk = {}
            else:
                with open(json_path, "r") as file:
                    dict_ldmk = json.load(file)
            # print(context_video_path, nb_frame_to_keep, dict_ldmk)
            frames = get_frames(context_video_path,
                                nb_frame_to_keep, dict_ldmk)
            # print(frames)
            ldmks = get_ldmk(frames)

            dict_ldmk = {}
            for index, ldk in ldmks:
                dict_ldmk.update({index: ldk.tolist()})

            with open(json_path, 'w') as file:
                json.dump(dict_ldmk, file, sort_keys=True)


if __name__ == "__main__":
    args = parse_args()
    process(args.global_video_path, args.global_image_path, args.total_frame_nb,
            args.from_i, args.to_i)
