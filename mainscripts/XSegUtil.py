import json
import shutil
import sys
import traceback
from pathlib import Path

import numpy as np

from core import pathex
from core.cv2ex import *
from core.interact import interact as io
from core.leras import nn
from core.leras.checkpoint import CheckpointLoadError
from DFLIMG import *
from facelib import XSegNet, LandmarksProcessor, FaceType
import pickle


# --- Generic XSeg model resource contract (P13-GENERIC-XSEG-RESOURCE-POLICY)
#
# This distribution bundles no XSeg model bytes of any kind and performs no
# network downloads. `xseg apply` requires the caller to point --model-dir
# at a model directory the user has obtained and verified. The documented,
# curated-launcher location for a *generic* (pretrained, i.e. not the
# user's own training result) model is:
#
#     resources/xseg_generic_model        (relative to the repository root)
#
# It is a host-local, untracked, user-managed directory: this repository
# never tracks, bundles, or redistributes model bytes there. Historical
# DeepFaceLab distributions shipped a generic pack (e.g. under
# `_internal\model_generic_xseg`); its provenance is unverified and it is
# intentionally absent from this repository. See the curated
# xseg-apply-generic-masks-* launcher headers for the user-facing policy.
#
# Directory contract (structural preflight validates the layout; the
# loader then strictly validates file contents itself):
#   XSeg_256.npy   required - model weights
#   XSeg_data.dat  optional - face-type metadata; when absent (or
#                     unreadable) the CLI asks for the face type
#                     interactively
XSEG_MODEL_REQUIRED_FILE = 'XSeg_256.npy'
XSEG_MODEL_OPTIONAL_FILE = 'XSeg_data.dat'
XSEG_GENERIC_MODEL_LOCATION = 'resources/xseg_generic_model'


def _repo_root():
    # mainscripts/XSegUtil.py -> repository root
    return Path(__file__).resolve().parents[1]


def _is_documented_generic_location(model_path):
    """True when --model-dir points at the documented generic-model
    location, in any relative form or as an absolute path."""
    try:
        normalized = str(model_path).replace('\\', '/').strip()
        while normalized.endswith('/') or normalized.endswith('\\'):
            normalized = normalized[:-1]
        if normalized.lower() == XSEG_GENERIC_MODEL_LOCATION:
            return True
        try:
            return Path(model_path).resolve() == (
                _repo_root() / 'resources' / 'xseg_generic_model')
        except Exception:
            return False
    except Exception:
        return False


def _documented_location_note(model_path):
    """Actionable guidance for the documented generic-model location,
    shown only when the failed --model-dir is that location."""
    if not _is_documented_generic_location(model_path):
        return []
    return [
        "",
        "That path is the documented, host-local location for a USER-PROVIDED",
        "generic XSeg model (used by the curated xseg-apply-generic-masks-*",
        "launchers). It is untracked and never shipped by this repository: no",
        "historical DeepFaceLab pack is bundled or redistributed here, and this",
        "distribution performs no downloads (provenance of historical packs is",
        "unverified). Place a compatible model directory you have verified",
        "yourself at that location, or run the CLI with --model-dir pointing",
        "at any other compatible model directory (e.g. your own trained XSeg",
        "model).",
    ]


def _fail_with(lines):
    """Print every line as an error-log line and exit with code 1.

    The chosen failure style for resource-policy violations: an
    actionable diagnostic (never a raw FileNotFoundError / NoneType
    traceback), a nonzero exit code (never a silent success or skip),
    and no fallback to any other model location.
    """
    for line in lines:
        io.log_err(line)
    sys.exit(1)


def _validate_apply_resources(input_path, model_path):
    """Structural preflight for `xseg apply`.

    Runs before any interactive prompt, device selection, NN
    initialization, or model loading, so a missing or structurally
    invalid resource stops with an actionable diagnostic. The loader
    still validates file contents afterwards (a corrupt model file ->
    classified failure, see apply_xseg).
    """
    if not input_path.exists():
        _fail_with([
            f"ERROR: input directory not found: {input_path}",
            "",
            "Pass --input-dir of an existing directory containing aligned",
            "face images (DFLIMG). Example:",
            '  dfl.bat xseg apply --input-dir "workspace\\data_src\\aligned" --model-dir <model dir>',
        ])
    if not input_path.is_dir():
        _fail_with([
            f"ERROR: --input-dir is a file, not a directory: {input_path}",
            "",
            "Pass --input-dir of an existing DIRECTORY containing aligned",
            "face images (DFLIMG).",
        ])
    if not model_path.exists():
        _fail_with([
            f"ERROR: XSeg model directory not found: {model_path}",
            "",
            "A valid --model-dir must be an EXISTING DIRECTORY containing:",
            f"  {XSEG_MODEL_REQUIRED_FILE}   (required) model weights",
            f"  {XSEG_MODEL_OPTIONAL_FILE}   (optional) face-type metadata; without it the CLI asks interactively",
            "",
            "This distribution bundles no XSeg model and downloads nothing;",
            "--model-dir must point at a model directory you have obtained yourself.",
        ] + _documented_location_note(model_path))
    if not model_path.is_dir():
        _fail_with([
            f"ERROR: XSeg model directory is a file, not a directory: {model_path}",
            "",
            f"A valid --model-dir must be an EXISTING DIRECTORY containing {XSEG_MODEL_REQUIRED_FILE}",
            f"(and optionally {XSEG_MODEL_OPTIONAL_FILE}).",
        ])
    if not (model_path / XSEG_MODEL_REQUIRED_FILE).is_file():
        try:
            present = ', '.join(sorted(p.name for p in model_path.iterdir()))
        except OSError:
            present = '<unreadable directory>'
        _fail_with([
            f"ERROR: XSeg model directory is structurally incomplete: {model_path}",
            "",
            f"  required file missing: {XSEG_MODEL_REQUIRED_FILE}",
            f"  present files: {present or '(none)'}",
            "",
            "The directory exists but is not a usable XSeg model directory",
            "(expected layout listed above).",
        ] + _documented_location_note(model_path))


def _loader_failure_lines(model_path, error):
    """Actionable diagnostic for a model file that passed the structural
    preflight but whose contents fail to load. Only the typed resource/
    content failures the strict loader can raise are ever handed in here
    (CheckpointLoadError, pickle.UnpicklingError, EOFError); every other
    exception type propagates under the app's normal error policy and is
    never rewritten into a resource diagnostic."""
    lines = [
        f"ERROR: XSeg model could not be loaded from: {model_path}",
        "",
    ]
    if isinstance(error, CheckpointLoadError):
        lines += [
            "  the model file exists but FAILED strict weight validation",
            "  (likely a corrupt, truncated, or incompatible model file).",
        ]
    else:
        # pickle.UnpicklingError / EOFError: the bytes cannot be decoded
        # as a checkpoint at all (empty, truncated, or not a model file).
        lines += [
            "  the model file exists but is not a readable checkpoint",
            "  (its bytes are corrupt, truncated, or empty).",
        ]
    lines += [
        "",
        "  Re-obtain the model file from your source and verify its",
        "  integrity, or point --model-dir at a known-good compatible",
        "  model directory.",
        "",
        "This distribution bundles no XSeg model and downloads nothing;",
        "--model-dir must point at a model directory you have obtained yourself.",
    ]
    lines += _documented_location_note(model_path)
    return lines


def apply_xseg(input_path, model_path):
    io.log_info(f'Input directory: {input_path}')
    io.log_info(f'Model directory: {model_path}')

    # Structural preflight BEFORE any interactive prompt, device
    # selection, NN initialization, or model loading
    # (P13-GENERIC-XSEG-RESOURCE-POLICY): missing or structurally
    # invalid resources fail with actionable diagnostics - never a raw
    # loader exception, and never a silent fallback to another model
    # location.
    _validate_apply_resources(input_path, model_path)


    face_type = None
    
    model_dat = model_path / XSEG_MODEL_OPTIONAL_FILE
    if model_dat.exists():
        dat = None
        try:
            dat = pickle.loads( model_dat.read_bytes() )
        except Exception:
            io.log_err(f'WARNING: {model_dat} exists but could not be read; ignoring it and asking for the face type interactively.')
        if dat is not None:
            dat_options = dat.get('options', None) if isinstance(dat, dict) else None
            if dat_options is not None:
                face_type = dat_options.get('face_type', None)
        
        
        
    if face_type is None:
        face_type = io.input_str ("XSeg model face type", 'same', ['h','mf','f','wf','head','same'], help_message="Specify face type of trained XSeg model. For example if XSeg model trained as WF, but faceset is HEAD, specify WF to apply xseg only on WF part of HEAD. Default is 'same'").lower()
        if face_type == 'same':
            face_type = None
    
    if face_type is not None:
        face_type = {'h'  : FaceType.HALF,
                     'mf' : FaceType.MID_FULL,
                     'f'  : FaceType.FULL,
                     'wf' : FaceType.WHOLE_FACE,
                     'head' : FaceType.HEAD}[face_type]
                     
    io.log_info(f'Applying XSeg model to {input_path.name}/ folder.')

    device_config = nn.DeviceConfig.ask_choose_device(choose_only_one=True)
    nn.initialize(device_config)
        
    
    
    # The preflight already validated this directory; re-verify the
    # required weight file immediately before load so the known
    # "file removed after preflight" case stays actionable. This is a
    # state check, not exception-string classification.
    required_weight_file = model_path / XSEG_MODEL_REQUIRED_FILE
    if not required_weight_file.is_file():
        _fail_with([
            f"ERROR: XSeg model weight file is missing: {required_weight_file}",
            "",
            "  the model directory was structurally valid earlier in this",
            "  run, but the required file is no longer present:",
            f"    {XSEG_MODEL_REQUIRED_FILE}   (required)",
            f"    {XSEG_MODEL_OPTIONAL_FILE}   (optional)",
            "",
            "  Restore the file and re-run.",
            "",
            "This distribution bundles no XSeg model and downloads nothing;",
            "--model-dir must point at a model directory you have obtained yourself.",
        ] + _documented_location_note(model_path))

    try:
        xseg = XSegNet(name='XSeg',
                        load_weights=True,
                        weights_file_root=model_path,
                        data_format=nn.data_format,
                        raise_on_no_model_files=True)
    except (CheckpointLoadError, pickle.UnpicklingError, EOFError) as e:
        # Known resource/content failures of the strict loader only:
        # CheckpointLoadError (the file exists but failed strict weight
        # validation) and pickle.UnpicklingError / EOFError (the bytes
        # cannot be decoded as a checkpoint at all - unpicklable, empty,
        # or truncated; the loader's contract is that an existing file
        # that cannot be loaded STRICTLY fails, and unpicklable bytes are
        # the one content failure it surfaces without its typed
        # CheckpointLoadError). Nothing else is caught here: device,
        # implementation, and runtime errors propagate under the app's
        # normal error policy and are never rewritten into a resource
        # diagnostic.
        _fail_with(_loader_failure_lines(model_path, e))
    xseg_res = xseg.get_resolution()
              
    images_paths = pathex.get_image_paths(input_path, return_Path_class=True)
    
    for filepath in io.progress_bar_generator(images_paths, "Processing"):
        dflimg = DFLIMG.load(filepath)
        if dflimg is None or not dflimg.has_data():
            io.log_info(f'{filepath} is not a DFLIMG')
            continue
        
        img = cv2_imread(filepath).astype(np.float32) / 255.0
        h,w,c = img.shape
        
        img_face_type = FaceType.fromString( dflimg.get_face_type() )
        if face_type is not None and img_face_type != face_type:
            lmrks = dflimg.get_source_landmarks()
            
            fmat = LandmarksProcessor.get_transform_mat(lmrks, w, face_type)
            imat = LandmarksProcessor.get_transform_mat(lmrks, w, img_face_type)
            
            g_p = LandmarksProcessor.transform_points (np.float32([(0,0),(w,0),(0,w) ]), fmat, True)
            g_p2 = LandmarksProcessor.transform_points (g_p, imat)
            
            mat = cv2.getAffineTransform( g_p2, np.float32([(0,0),(w,0),(0,w) ]) )
            
            img = cv2.warpAffine(img, mat, (w, w), cv2.INTER_LANCZOS4)
            img = cv2.resize(img, (xseg_res, xseg_res), interpolation=cv2.INTER_LANCZOS4)
        else:
            if w != xseg_res:
                img = cv2.resize( img, (xseg_res,xseg_res), interpolation=cv2.INTER_LANCZOS4 )    
                    
        if len(img.shape) == 2:
            img = img[...,None]            
    
        mask = xseg.extract(img)
        
        if face_type is not None and img_face_type != face_type:
            mask = cv2.resize(mask, (w, w), interpolation=cv2.INTER_LANCZOS4)
            mask = cv2.warpAffine( mask, mat, (w,w), np.zeros( (h,w,c), dtype=np.float), cv2.WARP_INVERSE_MAP | cv2.INTER_LANCZOS4)
            mask = cv2.resize(mask, (xseg_res, xseg_res), interpolation=cv2.INTER_LANCZOS4)
        mask[mask < 0.5]=0
        mask[mask >= 0.5]=1    
        dflimg.set_xseg_mask(mask)
        dflimg.save()


        
def fetch_xseg(input_path):
    if not input_path.exists():
        raise ValueError(f'{input_path} not found. Please ensure it exists.')
    
    output_path = input_path.parent / (input_path.name + '_xseg')
    output_path.mkdir(exist_ok=True, parents=True)
    
    io.log_info(f'Copying faces containing XSeg polygons to {output_path.name}/ folder.')
    
    images_paths = pathex.get_image_paths(input_path, return_Path_class=True)
    
    
    files_copied = []
    for filepath in io.progress_bar_generator(images_paths, "Processing"):
        dflimg = DFLIMG.load(filepath)
        if dflimg is None or not dflimg.has_data():
            io.log_info(f'{filepath} is not a DFLIMG')
            continue
        
        ie_polys = dflimg.get_seg_ie_polys()

        if ie_polys.has_polys():
            files_copied.append(filepath)
            shutil.copy ( str(filepath), str(output_path / filepath.name) )
    
    io.log_info(f'Files copied: {len(files_copied)}')
    
    is_delete = io.input_bool (f"\r\nDelete original files?", True)
    if is_delete:
        for filepath in files_copied:
            Path(filepath).unlink()
            
    
def remove_xseg(input_path):
    if not input_path.exists():
        raise ValueError(f'{input_path} not found. Please ensure it exists.')
    
    io.log_info(f'Processing folder {input_path}')
    io.log_info('!!! WARNING : APPLIED XSEG MASKS WILL BE REMOVED FROM THE FRAMES !!!')
    io.log_info('!!! WARNING : APPLIED XSEG MASKS WILL BE REMOVED FROM THE FRAMES !!!')
    io.log_info('!!! WARNING : APPLIED XSEG MASKS WILL BE REMOVED FROM THE FRAMES !!!')
    io.input_str('Press enter to continue.')
                               
    images_paths = pathex.get_image_paths(input_path, return_Path_class=True)
    
    files_processed = 0
    for filepath in io.progress_bar_generator(images_paths, "Processing"):
        dflimg = DFLIMG.load(filepath)
        if dflimg is None or not dflimg.has_data():
            io.log_info(f'{filepath} is not a DFLIMG')
            continue
        
        if dflimg.has_xseg_mask():
            dflimg.set_xseg_mask(None)
            dflimg.save()
            files_processed += 1
    io.log_info(f'Files processed: {files_processed}')
    
def remove_xseg_labels(input_path):
    if not input_path.exists():
        raise ValueError(f'{input_path} not found. Please ensure it exists.')
    
    io.log_info(f'Processing folder {input_path}')
    io.log_info('!!! WARNING : LABELED XSEG POLYGONS WILL BE REMOVED FROM THE FRAMES !!!')
    io.log_info('!!! WARNING : LABELED XSEG POLYGONS WILL BE REMOVED FROM THE FRAMES !!!')
    io.log_info('!!! WARNING : LABELED XSEG POLYGONS WILL BE REMOVED FROM THE FRAMES !!!')
    io.input_str('Press enter to continue.')
    
    images_paths = pathex.get_image_paths(input_path, return_Path_class=True)
    
    files_processed = 0
    for filepath in io.progress_bar_generator(images_paths, "Processing"):
        dflimg = DFLIMG.load(filepath)
        if dflimg is None or not dflimg.has_data():
            io.log_info(f'{filepath} is not a DFLIMG')
            continue

        if dflimg.has_seg_ie_polys():
            dflimg.set_seg_ie_polys(None)
            dflimg.save()            
            files_processed += 1
            
    io.log_info(f'Files processed: {files_processed}')