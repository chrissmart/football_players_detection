from django.shortcuts import render
from django.http import HttpResponse

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys
sys.path.append(r"C:\Users\chris\Google 雲端硬碟\trabajo\1. ml_ai_devp_house\2_technical_activity\django_project\football_players_detection\football_players_detection\football_players_detection")
from settings import STATICFILES_DIRS

# image processing packages
import cv2 as cv
import face_recognition
import youtube_dl
# Standard PySceneDetect imports:
from scenedetect.video_manager import VideoManager
from scenedetect.scene_manager import SceneManager
# For caching detection metrics and saving/loading to a stats file
from scenedetect.stats_manager import StatsManager
# For content-aware scene detection:
from scenedetect.detectors.content_detector import ContentDetector


# Create your views here.
def index(request):

    """
    :param request:
    :return:
    """

    """
    Section 1: Images Processing
    """
    # convert target image files into matrices
    image_to_be_searched_file = os.listdir(STATICFILES_DIRS[0] + "images")  # set images directories to be searched to a list
    image_to_be_searched_list = convert_image_to_matrix(image_to_be_searched_file)

    # extract and Save Target Image Feature Vectors
    target_image_locations_list = []
    target_image_embeddings_list = []
    for image_matrix in image_to_be_searched_list:
        face_locations, face_embeddings = face_detection_embeddings_model(image_matrix)
        target_image_locations_list.append(face_locations)  # (top, right, bottom, left)
        target_image_embeddings_list.append(face_embeddings)  # dimension = 128

    print(np.array(target_image_locations_list).shape)
    print(np.array(target_image_embeddings_list).shape)


    """
    Section 2: Video Processing
    """
    # download the video
    if request.POST:
        video_link = request.POST['q'] # https://www.youtube.com/watch?v=LYrFkaMx6e4

    video_download(download_video_or_not = 0, video_link = video_link)

    # scene detection for the appointed video
    print(os.listdir())
    video_path = r'10 Best Barcelona Players Of All Time-LYrFkaMx6e4.mp4'
    scene_list = find_scenes(video_path)


    return render(request, 'face_detection/index.html')


def convert_image_to_matrix(image_to_be_searched_file):
    image_to_be_searched_list = []
    for image in image_to_be_searched_file:
        # print(image)
        image_matrix = face_recognition.load_image_file(STATICFILES_DIRS[0] + "images\\" + image)
        image_to_be_searched_list.append(image_matrix)
        plt.imshow(image_matrix)
        plt.show()

    print("image_matrix.shape ", image_matrix.shape)

    return image_to_be_searched_list

def video_download(download_video_or_not = 1, video_link = 'https://www.youtube.com/watch?v=LYrFkaMx6e4'):
    # video used to search for the players
    if download_video_or_not == 1:
        video_list = [video_link]

        ydl_opts = {}
        for video in video_list:
            with youtube_dl.YoutubeDL(ydl_opts) as ydl:
                ydl.download([video])


def face_detection_embeddings_model(image_matrix):
    """
    face detector: Histogram Oriented Gradient features (caclulate occurrences of gradient orientation in localized areas of an image)
                   + linear SVM classifier
    face embeddings model: ResNet-34 classification model trained on 3 million faces
    """
    face_locations = face_recognition.face_locations(image_matrix)  # face detection model: (top, right, bottom, left)

    # if there is only one face in the image
    if len(face_locations) == 1:
        face_embeddings = face_recognition.face_encodings(image_matrix,
                                                          face_locations)  # face embeddings model, dimension = 128
        return face_locations, face_embeddings

    # if there are more than one face in the image
    elif len(face_locations) > 1:
        face_embeddings = []
        for face in face_locations:
            face_embeddings.append(face_recognition.face_encodings(image_matrix, face_locations))
        return face_locations, face_embeddings

    # no face detected
    else:
        return [], []


def find_scenes(video_path):
    """
    based on changes between frames in the HSV color space
    """
    # three main calsses: VideoManager, SceneManager, StatsManager
    video_manager = VideoManager([video_path])
    stats_manager = StatsManager()
    scene_manager = SceneManager(stats_manager)

    # Add ContentDetector algorithm (each detector's constructor
    # takes detector options, e.g. threshold).
    scene_manager.add_detector(ContentDetector())

    base_timecode = video_manager.get_base_timecode()

    # We save our stats file to {VIDEO_PATH}.stats.csv.
    stats_file_path = '%s.stats.csv' % video_path

    scene_list = []

    try:
        # If stats file exists, load it.
        if os.path.exists(stats_file_path):
            # Read stats from CSV file opened in read mode:
            with open(stats_file_path, 'r') as stats_file:
                stats_manager.load_from_csv(stats_file, base_timecode)

        # Set downscale factor to improve processing speed.
        video_manager.set_downscale_factor()

        # Start video_manager.
        video_manager.start()

        # Perform scene detection on video_manager.
        scene_manager.detect_scenes(frame_source=video_manager)

        # Obtain list of detected scenes.
        scene_list = scene_manager.get_scene_list(base_timecode)
        # Each scene is a tuple of (start, end) FrameTimecodes.

        print('List of scenes obtained:')
        for i, scene in enumerate(scene_list):
            print(
                'Scene %2d: Start %s / Frame %d, End %s / Frame %d' % (
                    i + 1,
                    scene[0].get_timecode(), scene[0].get_frames(),
                    scene[1].get_timecode(), scene[1].get_frames(),))

        # We only write to the stats file if a save is required:
        if stats_manager.is_save_required():
            with open(stats_file_path, 'w') as stats_file:
                stats_manager.save_to_csv(stats_file, base_timecode)

    finally:
        video_manager.release()

    return scene_list