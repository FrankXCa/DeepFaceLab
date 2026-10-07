import math
import multiprocessing
import operator
import os
import sys
import tempfile
import traceback
from functools import cmp_to_key
from pathlib import Path

import cv2
import numpy as np
from numpy import linalg as npla

from core import imagelib, mathlib, pathex
from core.cv2ex import *
from core.imagelib import estimate_sharpness
from core.interact import interact as io
from core.joblib import Subprocessor
from core.leras import nn
from DFLIMG import *
from facelib import LandmarksProcessor


FATAL_SCORER_FAILURE = "FATAL_SCORER_FAILURE"


class SharpnessScoringError(RuntimeError):
    """A blur scorer failed; sorter output must remain untouched."""


def _fatal_scorer_result(filepath, error):
    return [
        FATAL_SCORER_FAILURE,
        str(filepath),
        type(error).__name__,
        str(error),
        traceback.format_exc(),
    ]


def _raise_fatal_scorer_errors(errors):
    details = []
    for result in errors:
        _, pathname, error_type, message, diagnostic = result
        details.append(
            f"{pathname}: {error_type}: {message}\n{diagnostic}".rstrip())
    raise SharpnessScoringError(
        "FATAL_SCORER_FAILURE: blur sharpness scoring failed:\n" +
        "\n".join(details))


class BlurEstimatorSubprocessor(Subprocessor):
    class Cli(Subprocessor.Cli):
        def on_initialize(self, client_dict):
            self.estimate_motion_blur = client_dict['estimate_motion_blur']
        
        #override
        def process_data(self, data):
            filepath = Path( data[0] )
            dflimg = DFLIMG.load (filepath)

            if dflimg is None or not dflimg.has_data():
                self.log_err (f"{filepath.name} is not a dfl image file")
                return [ str(filepath), 0 ]
            else:
                try:
                    image = cv2_imread( str(filepath) )
                    if image is None:
                        raise ValueError(f"Unable to decode {filepath.name}")

                    face_mask = LandmarksProcessor.get_image_hull_mask (
                        image.shape, dflimg.get_landmarks())
                    image = (image*face_mask).astype(np.uint8)

                    if self.estimate_motion_blur:
                        value = cv2.Laplacian(
                            image, cv2.CV_64F, ksize=11).var()
                    else:
                        value = estimate_sharpness(image)
                except Exception as error:
                    return _fatal_scorer_result(filepath, error)

                return [ str(filepath), value ]


        #override
        def get_data_name (self, data):
            #return string identificator of your data
            return data[0]

    #override
    def __init__(self, input_data, estimate_motion_blur=False ):
        self.input_data = input_data
        self.estimate_motion_blur = estimate_motion_blur
        self.img_list = []
        self.trash_img_list = []
        self.fatal_errors = []
        super().__init__('BlurEstimator', BlurEstimatorSubprocessor.Cli, 60)

    #override
    def on_clients_initialized(self):
        io.progress_bar ("", len (self.input_data))

    #override
    def on_clients_finalized(self):
        io.progress_bar_close ()

    #override
    def process_info_generator(self):
        cpu_count = multiprocessing.cpu_count()
        io.log_info(f'Running on {cpu_count} CPUs')

        for i in range(cpu_count):
            yield 'CPU%d' % (i), {}, {'estimate_motion_blur':self.estimate_motion_blur}

    #override
    def get_data(self, host_dict):
        if self.fatal_errors:
            return None
        if len (self.input_data) > 0:
            return self.input_data.pop(0)

        return None

    #override
    def on_data_return (self, host_dict, data):
        self.input_data.insert(0, data)

    #override
    def on_result (self, host_dict, data, result):
        if result[0] == FATAL_SCORER_FAILURE:
            self.fatal_errors.append(result)
        elif result[1] == 0:
            self.trash_img_list.append ( result )
        else:
            self.img_list.append ( result )

        io.progress_bar_inc(1)

    #override
    def get_result(self):
        if self.fatal_errors:
            _raise_fatal_scorer_errors(self.fatal_errors)
        return self.img_list, self.trash_img_list


def sort_by_blur(input_path):
    io.log_info ("Sorting by blur...")

    img_list = [ (filename,[]) for filename in pathex.get_image_paths(input_path) ]
    img_list, trash_img_list = BlurEstimatorSubprocessor (img_list).run()

    io.log_info ("Sorting...")
    img_list = sorted(img_list, key=operator.itemgetter(1), reverse=True)

    return img_list, trash_img_list
    
def sort_by_motion_blur(input_path):
    io.log_info ("Sorting by motion blur...")

    img_list = [ (filename,[]) for filename in pathex.get_image_paths(input_path) ]
    img_list, trash_img_list = BlurEstimatorSubprocessor (img_list, estimate_motion_blur=True).run()

    io.log_info ("Sorting...")
    img_list = sorted(img_list, key=operator.itemgetter(1), reverse=True)

    return img_list, trash_img_list
    
def sort_by_face_yaw(input_path):
    io.log_info ("Sorting by face yaw...")
    img_list = []
    trash_img_list = []
    for filepath in io.progress_bar_generator( pathex.get_image_paths(input_path), "Loading"):
        filepath = Path(filepath)

        dflimg = DFLIMG.load (filepath)

        if dflimg is None or not dflimg.has_data():
            io.log_err (f"{filepath.name} is not a dfl image file")
            trash_img_list.append ( [str(filepath)] )
            continue

        pitch, yaw, roll = LandmarksProcessor.estimate_pitch_yaw_roll ( dflimg.get_landmarks(), size=dflimg.get_shape()[1] )

        img_list.append( [str(filepath), yaw ] )

    io.log_info ("Sorting...")
    img_list = sorted(img_list, key=operator.itemgetter(1), reverse=True)

    return img_list, trash_img_list

def sort_by_face_pitch(input_path):
    io.log_info ("Sorting by face pitch...")
    img_list = []
    trash_img_list = []
    for filepath in io.progress_bar_generator( pathex.get_image_paths(input_path), "Loading"):
        filepath = Path(filepath)

        dflimg = DFLIMG.load (filepath)

        if dflimg is None or not dflimg.has_data():
            io.log_err (f"{filepath.name} is not a dfl image file")
            trash_img_list.append ( [str(filepath)] )
            continue

        pitch, yaw, roll = LandmarksProcessor.estimate_pitch_yaw_roll ( dflimg.get_landmarks(), size=dflimg.get_shape()[1] )

        img_list.append( [str(filepath), pitch ] )

    io.log_info ("Sorting...")
    img_list = sorted(img_list, key=operator.itemgetter(1), reverse=True)

    return img_list, trash_img_list

def sort_by_face_source_rect_size(input_path):
    io.log_info ("Sorting by face rect size...")
    img_list = []
    trash_img_list = []
    for filepath in io.progress_bar_generator( pathex.get_image_paths(input_path), "Loading"):
        filepath = Path(filepath)

        dflimg = DFLIMG.load (filepath)

        if dflimg is None or not dflimg.has_data():
            io.log_err (f"{filepath.name} is not a dfl image file")
            trash_img_list.append ( [str(filepath)] )
            continue

        source_rect = dflimg.get_source_rect()
        rect_area = mathlib.polygon_area(np.array(source_rect[[0,2,2,0]]).astype(np.float32), np.array(source_rect[[1,1,3,3]]).astype(np.float32))

        img_list.append( [str(filepath), rect_area ] )

    io.log_info ("Sorting...")
    img_list = sorted(img_list, key=operator.itemgetter(1), reverse=True)

    return img_list, trash_img_list



class HistSsimSubprocessor(Subprocessor):
    class Cli(Subprocessor.Cli):
        #override
        def process_data(self, data):
            img_list = []
            for x in data:
                img = cv2_imread(x)
                img_list.append ([x, cv2.calcHist([img], [0], None, [256], [0, 256]),
                                     cv2.calcHist([img], [1], None, [256], [0, 256]),
                                     cv2.calcHist([img], [2], None, [256], [0, 256])
                                 ])

            img_list_len = len(img_list)
            for i in range(img_list_len-1):
                min_score = float("inf")
                j_min_score = i+1
                for j in range(i+1,len(img_list)):
                    score = cv2.compareHist(img_list[i][1], img_list[j][1], cv2.HISTCMP_BHATTACHARYYA) + \
                            cv2.compareHist(img_list[i][2], img_list[j][2], cv2.HISTCMP_BHATTACHARYYA) + \
                            cv2.compareHist(img_list[i][3], img_list[j][3], cv2.HISTCMP_BHATTACHARYYA)
                    if score < min_score:
                        min_score = score
                        j_min_score = j
                img_list[i+1], img_list[j_min_score] = img_list[j_min_score], img_list[i+1]

                self.progress_bar_inc(1)

            return img_list

        #override
        def get_data_name (self, data):
            return "Bunch of images"

    #override
    def __init__(self, img_list ):
        self.img_list = img_list
        self.img_list_len = len(img_list)

        slice_count = 20000
        sliced_count = self.img_list_len // slice_count

        if sliced_count > 12:
            sliced_count = 11.9
            slice_count = int(self.img_list_len / sliced_count)
            sliced_count = self.img_list_len // slice_count

        self.img_chunks_list = [ self.img_list[i*slice_count : (i+1)*slice_count] for i in range(sliced_count) ] + \
                               [ self.img_list[sliced_count*slice_count:] ]

        self.result = []
        super().__init__('HistSsim', HistSsimSubprocessor.Cli, 0)

    #override
    def process_info_generator(self):
        cpu_count = len(self.img_chunks_list)
        io.log_info(f'Running on {cpu_count} threads')
        for i in range(cpu_count):
            yield 'CPU%d' % (i), {'i':i}, {}

    #override
    def on_clients_initialized(self):
        io.progress_bar ("Sorting", len(self.img_list))
        io.progress_bar_inc(len(self.img_chunks_list))

    #override
    def on_clients_finalized(self):
        io.progress_bar_close()

    #override
    def get_data(self, host_dict):
        if len (self.img_chunks_list) > 0:
            return self.img_chunks_list.pop(0)
        return None

    #override
    def on_data_return (self, host_dict, data):
        raise Exception("Fail to process data. Decrease number of images and try again.")

    #override
    def on_result (self, host_dict, data, result):
        self.result += result
        return 0

    #override
    def get_result(self):
        return self.result

def sort_by_hist(input_path):
    io.log_info ("Sorting by histogram similarity...")
    img_list = HistSsimSubprocessor(pathex.get_image_paths(input_path)).run()
    return img_list, []

class HistDissimSubprocessor(Subprocessor):
    class Cli(Subprocessor.Cli):
        #override
        def on_initialize(self, client_dict):
            self.img_list = client_dict['img_list']
            self.img_list_len = len(self.img_list)

        #override
        def process_data(self, data):
            i = data[0]
            score_total = 0
            for j in range( 0, self.img_list_len):
                if i == j:
                    continue
                score_total += cv2.compareHist(self.img_list[i][1], self.img_list[j][1], cv2.HISTCMP_BHATTACHARYYA)

            return score_total

        #override
        def get_data_name (self, data):
            #return string identificator of your data
            return self.img_list[data[0]][0]

    #override
    def __init__(self, img_list ):
        self.img_list = img_list
        self.img_list_range = [i for i in range(0, len(img_list) )]
        self.result = []
        super().__init__('HistDissim', HistDissimSubprocessor.Cli, 60)

    #override
    def on_clients_initialized(self):
        io.progress_bar ("Sorting", len (self.img_list) )

    #override
    def on_clients_finalized(self):
        io.progress_bar_close()

    #override
    def process_info_generator(self):
        cpu_count = min(multiprocessing.cpu_count(), 8)
        io.log_info(f'Running on {cpu_count} CPUs')
        for i in range(cpu_count):
            yield 'CPU%d' % (i), {}, {'img_list' : self.img_list}

    #override
    def get_data(self, host_dict):
        if len (self.img_list_range) > 0:
            return [self.img_list_range.pop(0)]

        return None

    #override
    def on_data_return (self, host_dict, data):
        self.img_list_range.insert(0, data[0])

    #override
    def on_result (self, host_dict, data, result):
        self.img_list[data[0]][2] = result
        io.progress_bar_inc(1)

    #override
    def get_result(self):
        return self.img_list

def sort_by_hist_dissim(input_path):
    io.log_info ("Sorting by histogram dissimilarity...")

    img_list = []
    trash_img_list = []
    for filepath in io.progress_bar_generator( pathex.get_image_paths(input_path), "Loading"):
        filepath = Path(filepath)

        dflimg = DFLIMG.load (filepath)

        image = cv2_imread(str(filepath))

        if dflimg is not None and dflimg.has_data():
            face_mask = LandmarksProcessor.get_image_hull_mask (image.shape, dflimg.get_landmarks())
            image = (image*face_mask).astype(np.uint8)

        img_list.append ([str(filepath), cv2.calcHist([cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)], [0], None, [256], [0, 256]), 0 ])

    img_list = HistDissimSubprocessor(img_list).run()

    io.log_info ("Sorting...")
    img_list = sorted(img_list, key=operator.itemgetter(2), reverse=True)

    return img_list, trash_img_list

def sort_by_brightness(input_path):
    io.log_info ("Sorting by brightness...")
    img_list = [ [x, np.mean ( cv2.cvtColor(cv2_imread(x), cv2.COLOR_BGR2HSV)[...,2].flatten()  )] for x in io.progress_bar_generator( pathex.get_image_paths(input_path), "Loading") ]
    io.log_info ("Sorting...")
    img_list = sorted(img_list, key=operator.itemgetter(1), reverse=True)
    return img_list, []

def sort_by_hue(input_path):
    io.log_info ("Sorting by hue...")
    img_list = [ [x, np.mean ( cv2.cvtColor(cv2_imread(x), cv2.COLOR_BGR2HSV)[...,0].flatten()  )] for x in io.progress_bar_generator( pathex.get_image_paths(input_path), "Loading") ]
    io.log_info ("Sorting...")
    img_list = sorted(img_list, key=operator.itemgetter(1), reverse=True)
    return img_list, []

def sort_by_black(input_path):
    io.log_info ("Sorting by amount of black pixels...")

    img_list = []
    for x in io.progress_bar_generator( pathex.get_image_paths(input_path), "Loading"):
        img = cv2_imread(x)
        img_list.append ([x, img[(img == 0)].size ])

    io.log_info ("Sorting...")
    img_list = sorted(img_list, key=operator.itemgetter(1), reverse=False)

    return img_list, []

def sort_by_origname(input_path):
    io.log_info ("Sort by original filename...")

    img_list = []
    trash_img_list = []
    for filepath in io.progress_bar_generator( pathex.get_image_paths(input_path), "Loading"):
        filepath = Path(filepath)

        dflimg = DFLIMG.load (filepath)

        if dflimg is None or not dflimg.has_data():
            io.log_err (f"{filepath.name} is not a dfl image file")
            trash_img_list.append( [str(filepath)] )
            continue

        img_list.append( [str(filepath), dflimg.get_source_filename()] )

    io.log_info ("Sorting...")
    img_list = sorted(img_list, key=operator.itemgetter(1))
    return img_list, trash_img_list

def sort_by_oneface_in_image(input_path):
    io.log_info ("Sort by one face in images...")
    image_paths = pathex.get_image_paths(input_path)
    a = np.array ([ ( int(x[0]), int(x[1]) ) \
                      for x in [ Path(filepath).stem.split('_') for filepath in image_paths ] if len(x) == 2
                  ])
    if len(a) > 0:
        idxs = np.ndarray.flatten ( np.argwhere ( a[:,1] != 0 ) )
        idxs = np.unique ( a[idxs][:,0] )
        idxs = np.ndarray.flatten ( np.argwhere ( np.array([ x[0] in idxs for x in a ]) == True ) )
        if len(idxs) > 0:
            io.log_info ("Found %d images." % (len(idxs)) )
            img_list = [ (path,) for i,path in enumerate(image_paths) if i not in idxs ]
            trash_img_list = [ (image_paths[x],) for x in idxs ]
            return img_list, trash_img_list

    io.log_info ("Nothing found. Possible recover original filenames first.")
    return [], []

class FinalLoaderSubprocessor(Subprocessor):
    class Cli(Subprocessor.Cli):
        #override
        def on_initialize(self, client_dict):
            self.faster = client_dict['faster']

        #override
        def process_data(self, data):
            filepath = Path(data[0])

            try:
                dflimg = DFLIMG.load (filepath)

                if dflimg is None or not dflimg.has_data():
                    self.log_err (f"{filepath.name} is not a dfl image file")
                    return [ 1, [str(filepath)] ]

                bgr = cv2_imread(str(filepath))
                if bgr is None:
                    raise ValueError("Unable to decode %s" % filepath.name)

                gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            except Exception as error:
                if not self.faster:
                    return _fatal_scorer_result(filepath, error)
                self.log_err(error)
                return [ 1, [str(filepath)] ]

            try:
                if self.faster:
                    source_rect = dflimg.get_source_rect()
                    sharpness = mathlib.polygon_area(np.array(source_rect[[0,2,2,0]]).astype(np.float32), np.array(source_rect[[1,1,3,3]]).astype(np.float32))
                else:
                    face_mask = LandmarksProcessor.get_image_hull_mask (gray.shape, dflimg.get_landmarks())     
                    sharpness = estimate_sharpness( (gray[...,None]*face_mask).astype(np.uint8) )
            except Exception as error:
                if not self.faster:
                    return _fatal_scorer_result(filepath, error)
                self.log_err(error)
                return [ 1, [str(filepath)] ]

            try:
                pitch, yaw, roll = LandmarksProcessor.estimate_pitch_yaw_roll ( dflimg.get_landmarks(), size=dflimg.get_shape()[1] )

                hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            except Exception as e:
                self.log_err (e)
                return [ 1, [str(filepath)] ]

            return [ 0, [str(filepath), sharpness, hist, yaw, pitch ] ]

        #override
        def get_data_name (self, data):
            #return string identificator of your data
            return data[0]

    #override
    def __init__(self, img_list, faster ):
        self.img_list = img_list

        self.faster = faster
        self.result = []
        self.result_trash = []
        self.fatal_errors = []

        super().__init__('FinalLoader', FinalLoaderSubprocessor.Cli, 60)

    #override
    def on_clients_initialized(self):
        io.progress_bar ("Loading", len (self.img_list))

    #override
    def on_clients_finalized(self):
        io.progress_bar_close()

    #override
    def process_info_generator(self):
        cpu_count = min(multiprocessing.cpu_count(), 8)
        io.log_info(f'Running on {cpu_count} CPUs')

        for i in range(cpu_count):
            yield 'CPU%d' % (i), {}, {'faster': self.faster}

    #override
    def get_data(self, host_dict):
        if self.fatal_errors:
            return None
        if len (self.img_list) > 0:
            return [self.img_list.pop(0)]

        return None

    #override
    def on_data_return (self, host_dict, data):
        self.img_list.insert(0, data[0])

    #override
    def on_result (self, host_dict, data, result):
        if result[0] == FATAL_SCORER_FAILURE:
            self.fatal_errors.append(result)
        elif result[0] == 0:
            self.result.append (result[1])
        else:
            self.result_trash.append (result[1])
        io.progress_bar_inc(1)

    #override
    def get_result(self):
        if self.fatal_errors:
            _raise_fatal_scorer_errors(self.fatal_errors)
        return self.result, self.result_trash

class FinalHistDissimSubprocessor(Subprocessor):
    class Cli(Subprocessor.Cli):
        #override
        def process_data(self, data):
            idx, pitch_yaw_img_list = data

            for p in range ( len(pitch_yaw_img_list) ):

                img_list = pitch_yaw_img_list[p]
                if img_list is not None:
                    for i in range( len(img_list) ):
                        score_total = 0
                        for j in range( len(img_list) ):
                            if i == j:
                                continue
                            score_total += cv2.compareHist(img_list[i][2], img_list[j][2], cv2.HISTCMP_BHATTACHARYYA)
                        img_list[i][3] = score_total

                    pitch_yaw_img_list[p] = sorted(img_list, key=operator.itemgetter(3), reverse=True)

            return idx, pitch_yaw_img_list

        #override
        def get_data_name (self, data):
            return "Bunch of images"

    #override
    def __init__(self, pitch_yaw_sample_list ):
        self.pitch_yaw_sample_list = pitch_yaw_sample_list
        self.pitch_yaw_sample_list_len = len(pitch_yaw_sample_list)

        self.pitch_yaw_sample_list_idxs = [ i for i in range(self.pitch_yaw_sample_list_len) if self.pitch_yaw_sample_list[i] is not None ]
        self.result = [ None for _ in range(self.pitch_yaw_sample_list_len) ]
        super().__init__('FinalHistDissimSubprocessor', FinalHistDissimSubprocessor.Cli)

    #override
    def process_info_generator(self):
        cpu_count = min(multiprocessing.cpu_count(), 8)
        io.log_info(f'Running on {cpu_count} CPUs')
        for i in range(cpu_count):
            yield 'CPU%d' % (i), {}, {}

    #override
    def on_clients_initialized(self):
        io.progress_bar ("Sort by hist-dissim", len(self.pitch_yaw_sample_list_idxs) )

    #override
    def on_clients_finalized(self):
        io.progress_bar_close()

    #override
    def get_data(self, host_dict):
        if len (self.pitch_yaw_sample_list_idxs) > 0:
            idx = self.pitch_yaw_sample_list_idxs.pop(0)

            return idx, self.pitch_yaw_sample_list[idx]
        return None

    #override
    def on_data_return (self, host_dict, data):
        self.pitch_yaw_sample_list_idxs.insert(0, data[0])

    #override
    def on_result (self, host_dict, data, result):
        idx, yaws_sample_list = data
        self.result[idx] = yaws_sample_list
        io.progress_bar_inc(1)

    #override
    def get_result(self):
        return self.result

def sort_best_faster(input_path):
    return sort_best(input_path, faster=True)

def sort_best(input_path, faster=False):
    target_count = io.input_int ("Target number of faces?", 2000)

    io.log_info ("Performing sort by best faces.")
    if faster:
        io.log_info("Using faster algorithm. Faces will be sorted by source-rect-area instead of blur.")

    img_list, trash_img_list = FinalLoaderSubprocessor( pathex.get_image_paths(input_path), faster ).run()
    final_img_list = []

    grads = 128
    imgs_per_grad = round (target_count / grads)

    #instead of math.pi / 2, using -1.2,+1.2 because actually maximum yaw for 2DFAN landmarks are -1.2+1.2
    grads_space = np.linspace (-1.2, 1.2,grads)

    yaws_sample_list = [None]*grads
    for g in io.progress_bar_generator ( range(grads), "Sort by yaw"):
        yaw = grads_space[g]
        next_yaw = grads_space[g+1] if g < grads-1 else yaw

        yaw_samples = []
        for img in img_list:
            s_yaw = -img[3]
            if (g == 0          and s_yaw < next_yaw) or \
               (g < grads-1     and s_yaw >= yaw and s_yaw < next_yaw) or \
               (g == grads-1    and s_yaw >= yaw):
                yaw_samples += [ img ]
        if len(yaw_samples) > 0:
            yaws_sample_list[g] = yaw_samples

    total_lack = 0
    for g in io.progress_bar_generator ( range(grads), ""):
        img_list = yaws_sample_list[g]
        img_list_len = len(img_list) if img_list is not None else 0

        lack = imgs_per_grad - img_list_len
        total_lack += max(lack, 0)

    imgs_per_grad += total_lack // grads


    sharpned_imgs_per_grad = imgs_per_grad*10
    for g in io.progress_bar_generator ( range (grads), "Sort by blur"):
        img_list = yaws_sample_list[g]
        if img_list is None:
            continue

        img_list = sorted(img_list, key=operator.itemgetter(1), reverse=True)

        if len(img_list) > sharpned_imgs_per_grad:
            trash_img_list += img_list[sharpned_imgs_per_grad:]
            img_list = img_list[0:sharpned_imgs_per_grad]

        yaws_sample_list[g] = img_list


    yaw_pitch_sample_list = [None]*grads
    pitch_grads = imgs_per_grad

    for g in io.progress_bar_generator ( range (grads), "Sort by pitch"):
        img_list = yaws_sample_list[g]
        if img_list is None:
            continue

        pitch_sample_list = [None]*pitch_grads

        grads_space = np.linspace (-math.pi / 2,math.pi / 2, pitch_grads )

        for pg in range (pitch_grads):

            pitch = grads_space[pg]
            next_pitch = grads_space[pg+1] if pg < pitch_grads-1 else pitch

            pitch_samples = []
            for img in img_list:
                s_pitch = img[4]
                if (pg == 0                and s_pitch < next_pitch) or \
                   (pg < pitch_grads-1     and s_pitch >= pitch and s_pitch < next_pitch) or \
                   (pg == pitch_grads-1    and s_pitch >= pitch):
                    pitch_samples += [ img ]

            if len(pitch_samples) > 0:
                pitch_sample_list[pg] = pitch_samples
        yaw_pitch_sample_list[g] = pitch_sample_list

    yaw_pitch_sample_list = FinalHistDissimSubprocessor(yaw_pitch_sample_list).run()

    for g in io.progress_bar_generator (range (grads), "Fetching the best"):
        pitch_sample_list = yaw_pitch_sample_list[g]
        if pitch_sample_list is None:
            continue

        n = imgs_per_grad

        while n > 0:
            n_prev = n
            for pg in range(pitch_grads):
                img_list = pitch_sample_list[pg]
                if img_list is None:
                    continue
                final_img_list += [ img_list.pop(0) ]
                if len(img_list) == 0:
                    pitch_sample_list[pg] = None
                n -= 1
                if n == 0:
                    break
            if n_prev == n:
                break

        for pg in range(pitch_grads):
            img_list = pitch_sample_list[pg]
            if img_list is None:
                continue
            trash_img_list += img_list

    return final_img_list, trash_img_list

"""
def sort_by_vggface(input_path):
    io.log_info ("Sorting by face similarity using VGGFace model...")

    model = VGGFace()

    final_img_list = []
    trash_img_list = []

    image_paths = pathex.get_image_paths(input_path)
    img_list = [ (x,) for x in image_paths ]
    img_list_len = len(img_list)
    img_list_range = [*range(img_list_len)]

    feats = [None]*img_list_len
    for i in io.progress_bar_generator(img_list_range, "Loading"):
        img = cv2_imread( img_list[i][0] ).astype(np.float32)
        img = imagelib.normalize_channels (img, 3)
        img = cv2.resize (img, (224,224) )
        img = img[..., ::-1]
        img[..., 0] -= 93.5940
        img[..., 1] -= 104.7624
        img[..., 2] -= 129.1863
        feats[i] = model.predict( img[None,...] )[0]

    tmp = np.zeros( (img_list_len,) )
    float_inf = float("inf")
    for i in io.progress_bar_generator ( range(img_list_len-1), "Sorting" ):
        i_feat = feats[i]

        for j in img_list_range:
            tmp[j] = npla.norm(i_feat-feats[j]) if j >= i+1 else float_inf

        idx = np.argmin(tmp)

        img_list[i+1], img_list[idx] = img_list[idx], img_list[i+1]
        feats[i+1], feats[idx] = feats[idx], feats[i+1]

    return img_list, trash_img_list
"""

def sort_by_absdiff(input_path):
    io.log_info ("Sorting by absolute difference...")

    image_paths = pathex.get_image_paths(input_path)
    image_paths_len = len(image_paths)

    # Validate the complete input before creating temporary storage or allowing
    # final_process() to rename anything.  The historical TensorFlow
    # placeholders accepted arbitrary, but uniform, HWC shapes.
    _validate_absdiff_images(image_paths)

    if image_paths_len == 0:
        return [], []
    if image_paths_len == 1:
        return [(image_paths[0],)], []

    is_sim = io.input_bool ("Sort by similar?", True, help_message="Otherwise sort by dissimilar.")

    device_config = nn.DeviceConfig.ask_choose_device(choose_only_one=True)
    from core.leras.device import get_torch_device
    import torch

    device = (torch.device("cpu") if len(device_config.devices) == 0
              else get_torch_device(device_config.devices[0]))
    sorted_ids = _sort_absdiff_image_paths(
        image_paths, is_sim, device, torch=torch, block_size=512)

    img_list = [ (image_paths[x],) for x in sorted_ids]
    return img_list, []


def _read_absdiff_image(image_path, expected_shape=None):
    image = cv2_imread(image_path)
    name = Path(image_path).name
    if image is None:
        raise ValueError(f"Unable to decode absdiff input image: {name}")
    if image.ndim != 3:
        raise ValueError(
            f"absdiff input image must be HWC (rank 3): {name} has "
            f"shape {image.shape}")
    if expected_shape is not None and image.shape != expected_shape:
        raise ValueError(
            f"absdiff input images must have one common HWC shape: {name} "
            f"has {image.shape}, expected {expected_shape}")
    return np.ascontiguousarray(image)


def _validate_absdiff_images(image_paths):
    expected_shape = None
    for image_path in image_paths:
        image = _read_absdiff_image(image_path, expected_shape)
        if expected_shape is None:
            expected_shape = image.shape
    return expected_shape


def _score_absdiff_blocks(i_images, j_images, device, torch):
    """Return float32 scores with rows from j_images and columns from i_images."""
    i_array = np.stack(i_images)
    i_tensor = torch.as_tensor(i_array, dtype=torch.float32, device=device)
    rows = []
    with torch.inference_mode():
        for j_image in j_images:
            j_tensor = torch.as_tensor(
                j_image, dtype=torch.float32, device=device)
            row = torch.sum(
                torch.abs(i_tensor - j_tensor),
                dim=(1, 2, 3),
                dtype=torch.float32,
            )
            rows.append(row.detach().cpu().numpy())
    return np.asarray(rows, dtype=np.float32)


def _greedy_absdiff_order(score_matrix, is_sim):
    image_count = score_matrix.shape[0]
    if image_count == 0:
        return []

    next_id = 0
    sorted_ids = [next_id]
    for _ in io.progress_bar_generator(range(image_count - 1), "Sorting"):
        id_ar = np.concatenate(
            [score_matrix[:next_id, next_id],
             score_matrix[next_id, next_id:]])
        id_ar = np.argsort(id_ar)
        remaining = np.setdiff1d(id_ar, sorted_ids, True)
        next_id = int(remaining[0 if is_sim else -1])
        sorted_ids.append(next_id)
    return sorted_ids


def _sort_absdiff_image_paths(image_paths, is_sim, device, torch,
                              block_size=512):
    image_count = len(image_paths)
    if image_count <= 1:
        return list(range(image_count))
    if block_size <= 0:
        raise ValueError("absdiff block_size must be positive")

    expected_shape = _read_absdiff_image(image_paths[0]).shape
    temp_fd, temp_name = tempfile.mkstemp(
        prefix="dfl_absdiff_", suffix=".mmap")
    os.close(temp_fd)
    score_matrix = None
    progress_open = False
    try:
        score_matrix = np.memmap(
            temp_name, dtype=np.float32, mode="w+",
            shape=(image_count, image_count))
        score_matrix[:] = 0.0

        block_count = (image_count + block_size - 1) // block_size
        progress_count = block_count * (block_count + 1) // 2
        io.progress_bar("Computing", progress_count)
        progress_open = True

        for j in range(0, image_count, block_size):
            j_paths = image_paths[j:j + block_size]
            j_images = [
                _read_absdiff_image(path, expected_shape) for path in j_paths]
            j_end = j + len(j_images)

            for i in range(j, image_count, block_size):
                i_paths = image_paths[i:i + block_size]
                i_images = [
                    _read_absdiff_image(path, expected_shape)
                    for path in i_paths]
                i_end = i + len(i_images)

                scores = _score_absdiff_blocks(
                    i_images, j_images, device, torch)
                score_matrix[j:j_end, i:i_end] = scores
                score_matrix[i:i_end, j:j_end] = scores.T
                io.progress_bar_inc(1)

        io.progress_bar_close()
        progress_open = False
        score_matrix.flush()
        return _greedy_absdiff_order(score_matrix, is_sim)
    finally:
        try:
            if progress_open:
                io.progress_bar_close()
        finally:
            try:
                if score_matrix is not None:
                    score_matrix.flush()
            finally:
                if score_matrix is not None:
                    mmap = getattr(score_matrix, "_mmap", None)
                    if mmap is not None:
                        mmap.close()
                try:
                    Path(temp_name).unlink()
                except FileNotFoundError:
                    pass

def final_process(input_path, img_list, trash_img_list):
    if len(trash_img_list) != 0:
        parent_input_path = input_path.parent
        trash_path = parent_input_path / (input_path.stem + '_trash')
        trash_path.mkdir (exist_ok=True)

        io.log_info ("Trashing %d items to %s" % ( len(trash_img_list), str(trash_path) ) )

        for filename in pathex.get_image_paths(trash_path):
            Path(filename).unlink()

        for i in io.progress_bar_generator( range(len(trash_img_list)), "Moving trash", leave=False):
            src = Path (trash_img_list[i][0])
            dst = trash_path / src.name
            try:
                src.rename (dst)
            except:
                io.log_info ('fail to trashing %s' % (src.name) )

        io.log_info ("")

    if len(img_list) != 0:
        for i in io.progress_bar_generator( [*range(len(img_list))], "Renaming", leave=False):
            src = Path (img_list[i][0])
            dst = input_path / ('%.5d_%s' % (i, src.name ))
            try:
                src.rename (dst)
            except:
                io.log_info ('fail to rename %s' % (src.name) )

        for i in io.progress_bar_generator( [*range(len(img_list))], "Renaming"):
            src = Path (img_list[i][0])
            src = input_path / ('%.5d_%s' % (i, src.name))
            dst = input_path / ('%.5d%s' % (i, src.suffix))
            try:
                src.rename (dst)
            except:
                io.log_info ('fail to rename %s' % (src.name) )

sort_func_methods = {
    'blur':        ("blur", sort_by_blur),
    'motion-blur': ("motion_blur", sort_by_motion_blur),
    'face-yaw':    ("face yaw direction", sort_by_face_yaw),
    'face-pitch':  ("face pitch direction", sort_by_face_pitch),
    'face-source-rect-size' : ("face rect size in source image", sort_by_face_source_rect_size),
    'hist':        ("histogram similarity", sort_by_hist),
    'hist-dissim': ("histogram dissimilarity", sort_by_hist_dissim),
    'brightness':  ("brightness", sort_by_brightness),
    'hue':         ("hue", sort_by_hue),
    'black':       ("amount of black pixels", sort_by_black),
    'origname':    ("original filename", sort_by_origname),
    'oneface':     ("one face in image", sort_by_oneface_in_image),
    'absdiff':     ("absolute pixel difference", sort_by_absdiff),
    'final':       ("best faces", sort_best),
    'final-fast':  ("best faces faster", sort_best_faster),
}

def main (input_path, sort_by_method=None):
    io.log_info ("Running sort tool.\r\n")

    if sort_by_method is None:
        io.log_info(f"Choose sorting method:")

        key_list = list(sort_func_methods.keys())
        for i, key in enumerate(key_list):
            desc, func = sort_func_methods[key]
            io.log_info(f"[{i}] {desc}")

        io.log_info("")
        id = io.input_int("", 5, valid_list=[*range(len(key_list))] )

        sort_by_method = key_list[id]
    else:
        sort_by_method = sort_by_method.lower()

    desc, func = sort_func_methods[sort_by_method]
    img_list, trash_img_list = func(input_path)

    final_process (input_path, img_list, trash_img_list)
