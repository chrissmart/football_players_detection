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
    # print(os.listdir())
    video_path = r'10 Best Barcelona Players Of All Time-LYrFkaMx6e4.mp4'
    scene_list = find_scenes(video_path)
    # extract key scene
    scene_frames, scene_frames_time = extract_key_scene(scene_list)

    """
    Section 3: Image Search in the Video
    """
    if request.POST:
        print("Post Triggered")
        player_to_be_found_no = request.POST['player']
        image_to_be_found_ind = int(player_to_be_found_no)
        video_search(target_image_embeddings_list, scene_frames, scene_frames_time, image_to_be_found_ind)


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

def extract_key_scene(scene_list):
    # to save frame from scene detection
    scene_frames = []
    scene_frames_time = []

    # video capture
    video_path = "10 Best Barcelona Players Of All Time-LYrFkaMx6e4.mp4"
    vc = cv.VideoCapture(video_path)

    # setting of parameters to capture the specified frame
    time_length = 200.0  # length of video (seconds): 00:03:20.801
    fps = 29.969940  # frames per second
    print("Total Frames: ", time_length * fps)

    # capture the scene to test the similarity
    for scene in scene_list:
        frame_seq = scene[0].get_frames()  # ranging from 0 to total number of frames (= time_length*fps - 1)
        frame_time_code = scene[0].get_timecode()
        vc.set(1,
               frame_seq)  # flag CV_CAP_PROP_POS_FRAMES which is a 0-based index of the frame to be decoded/captured next

        # Read the next frame from the video. If you set frame 749 above then the code will return the last frame.
        is_capturing, frame = vc.read()

        # show the frame image
        frame = cv.cvtColor(frame,
                            cv.COLOR_BGR2RGB)  # convert the channel of image from BGR to RGB, dimension: (720, 1280, 3)
        scene_frames.append(frame)
        scene_frames_time.append(frame_time_code)
    # print("frame_seq ", frame_seq)
    #     print("frame time code: ", frame_time_code)
    #     plt.imshow(frame)
    #     plt.show()

    # When everything done, release the capture
    vc.release()
    cv.destroyAllWindows()

    return (scene_frames, scene_frames_time)


def face_distance(face_encodings, face_to_compare, MAX_DISTANCE=0.6):
    """
    face_encodings, face_to_compare are in the format of list array
    similarity of face_encodings and face_to_compare is conducted
    """
    if len(face_to_compare) == 0:
        return None, np.empty((0))

    else:
        face_encodings = face_encodings  # list
        face_to_compare = face_to_compare[0]  # list of arrays
        distances = np.round(np.linalg.norm(face_encodings - face_to_compare, axis=1), 4)  # euclidean distance
        #         print(distances)
        matched_face_ind = np.argmin(distances)
        return matched_face_ind, distances[matched_face_ind]


def face_detection_box(frame, location, name = None):
    top, right, bottom, left = location[0]
    color = (20, 120, 20)
    cv.rectangle(frame, (left, top), (right, bottom), color, 3)


def video_search(target_image_embeddings_list, scene_frames, scene_frames_time, image_to_be_found_ind):
    # access image embeddings of the selected player
    target_image_embeddings = np.array(target_image_embeddings_list)[image_to_be_found_ind]  # image one [image_index]

    """start searching"""
    if_player_found = False  # to identify if a player is found or not
    # capture the frames from scene detection to accelerate the searching time
    for frame, time_code in zip(scene_frames, scene_frames_time):
        video_image_location, video_image_embeddings = face_detection_embeddings_model(frame)

        # similarity testing
        if video_image_embeddings != []:
            MAX_DISTANCE = 0.605
            matched_face_ind, distance = face_distance(target_image_embeddings, video_image_embeddings, MAX_DISTANCE)

            if distance <= MAX_DISTANCE:
                if_player_found = True
                # visualization

                if len(video_image_location) > 1:
                    face_detection_box(frame, [video_image_location[matched_face_ind]])
                else:
                    face_detection_box(frame, video_image_location)
                plt.imshow(frame)  # image found in the video
                plt.savefig("static\\images_2\\image_found_in_video.jpg")
                break

    if if_player_found == False:
        print("The assigned player is not found in the video.")

    return frame