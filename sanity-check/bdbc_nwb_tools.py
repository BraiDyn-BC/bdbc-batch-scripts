from typing import Union, Literal, Tuple, Dict, Optional
from pathlib import Path
from collections import namedtuple as _namedtuple

import numpy as _np
import numpy.typing as _npt
import pandas as _pd

from pynwb import (
    NWBHDF5IO as _NWBHDFIO,
    NWBFile as _NWBFile,
)

# data structures

PathLike = Union[str, Path]


class Metadata(_namedtuple('Metadata', (
    'session_id',
    'session_description',
    'session_notes',
    'subject_id',
    'subject_DoB',
    'subject_age',
    'subject_sex',
))):
    pass


class Timebases(_namedtuple('Timebases', (
    'daq',
    'imaging',
    'videos',
))):
    pass


class NWBData(_namedtuple('NWBData', (
    'metadata',
    'timebase',
    'trials',
    'daq',
    'body_video_tracking',
    'face_video_tracking',
    'eye_video_tracking',
    'pupil_tracking',
    'rois',
    'roi_description',
))):
    pass


class NWBData_resting(_namedtuple('NWBData', (
    'metadata',
    'timebase',
    'daq',
    'body_video_tracking',
    'face_video_tracking',
    'eye_video_tracking',
    'pupil_tracking',
    'rois',
    'roi_description',
))):
    pass


class NWBData_sensory(_namedtuple('NWBData', (
    'metadata',
    'timebase',
    'trials',
    'rois',
    'roi_description',
))):
    pass


# I/O routines

def read_metadata(nwbfile: _NWBFile) -> Metadata:
    metadata = {
        'session_id': nwbfile.session_id,
        'session_description': nwbfile.session_description,
        'session_notes': nwbfile.notes,
        'subject_id': nwbfile.subject.subject_id,
        'subject_DoB': nwbfile.subject.date_of_birth.strftime('%Y-%m-%d'),
        'subject_age': nwbfile.subject.age,
        'subject_sex': nwbfile.subject.sex
    }
    return Metadata(**metadata)


def read_timebases(
    nwbfile: _NWBFile,
    downsampled: bool = True,
) -> Union[Timebases, _npt.NDArray]:
    imaging = _np.array(nwbfile.get_acquisition('widefield_blue').timestamps)
    if downsampled:
        return imaging
    else:
        daq = _np.array(nwbfile.get_acquisition('Humidity_raw').timestamps)
        if 'body_video' in nwbfile.acquisition.keys():
            videos = _np.array(nwbfile.get_acquisition('body_video').timestamps)
        else:
            videos = None
        return Timebases(
            daq=daq,
            imaging=imaging,
            videos=videos,
        )


def read_roi_dFF(nwbfile: _NWBFile) -> Tuple[_pd.DataFrame, Dict[str, str]]:
    dff_entry = nwbfile.get_processing_module('ophys').get_data_interface('DfOverF').get_roi_response_series('dFF')
    dff = _np.array(dff_entry.data)
    names = tuple(str(name) for name in _np.array(dff_entry.rois.table.get('roi_name')))
    descs = tuple(str(desc) for desc in _np.array(dff_entry.rois.table.get('roi_description')))
    roidescs = dict((name, desc) for name, desc in zip(names, descs))
    roidata = _pd.DataFrame(data=dff, columns=names)
    return roidata, roidescs


def read_trials(
    nwbfile: _NWBFile,
    downsampled: bool = True
) -> Optional[_pd.DataFrame]:
    if nwbfile.trials is None:
        return None
    if downsampled:
        entry = nwbfile.get_processing_module('downsampled').get_data_interface('trials')
    else:
        entry = nwbfile.get_time_intervals('trials')
    trials = entry.to_dataframe()
    trials.index.name = None
    return trials


def read_acquisition(
    nwbfile: _NWBFile,
    downsampled: bool = True,
    isSensory: bool = False
) -> Optional[_pd.DataFrame]:
    if downsampled:
        parent = nwbfile.get_processing_module('downsampled')
    else:
        parent = nwbfile.acquisition
    data = dict()
    for name in parent.data_interfaces.keys():
        if ('video_keypoints' in name) or (name in ('eye_position', 'pupil_tracking', 'trials')):
            continue
        column = name.lower().replace(' ', '_').replace('.', '').replace('-', '_').replace('_ds', '')
        data[column] = _np.array(parent.get_data_interface(name).data)

    # data-type conversion
    if not isSensory:
        for column in ('reward', 'state_lever', 'tone'):
            data[column] = (data[column] > 0)
        for column in ('state_task',):
            data[column] = data[column].astype(_np.int16)

    return _pd.DataFrame(data=data)


def read_video_tracking(
    nwbfile: _NWBFile,
    view: Literal['body', 'face', 'eye', 'pupil'] = 'body',
    downsampled: bool = True
) -> Optional[_pd.DataFrame]:
    if view in ('body', 'face', 'eye'):
        name = f"{view}_video_keypoints"
        if downsampled:
            parent = nwbfile.get_processing_module('downsampled')
        else:
            parent = nwbfile.get_processing_module('behavior')
        if name not in parent.data_interfaces:
            return None
        entries = parent.get_data_interface(name)
        data = dict()
        for kpt in entries.pose_estimation_series.keys():
            values = _np.array(entries.get_pose_estimation_series(kpt).data)
            data[kpt, 'x'] = values[:, 0]
            data[kpt, 'y'] = values[:, 1]
            if not downsampled:
                data[kpt, 'likelihood'] = _np.array(entries.get_pose_estimation_series(kpt).confidence)
    elif view == 'pupil':
        if downsampled:
            parent = nwbfile.get_processing_module('downsampled')
        else:
            parent = nwbfile.get_processing_module('behavior')
        if 'pupil_tracking' not in parent.data_interfaces:
            return None
        data = dict()
        data['diameter'] = _np.array(parent.get_data_interface('pupil_tracking').get_timeseries('diameter').data)
        data['center_x'] = _np.array(parent.get_data_interface('eye_position').get_spatial_series('center_x').data)
        data['center_y'] = _np.array(parent.get_data_interface('eye_position').get_spatial_series('center_y').data)
    else:
        raise ValueError(f"expected 'body', 'face', 'eye' or 'pupil', but got '{view}'")
    return _pd.DataFrame(data=data)


def load_from_file(
    nwbfilepath: PathLike,
    downsampled: bool = True,
    isResting: bool = False,
    isSensory: bool = False,
) -> NWBData:
    with _NWBHDFIO(nwbfilepath, mode='r') as src:
        nwbfile = src.read()
        data = dict()
        data['metadata'] = read_metadata(nwbfile)
        data['timebase'] = read_timebases(nwbfile, downsampled=downsampled)
        if not isResting:
            data['trials'] = read_trials(nwbfile, downsampled=downsampled)
        if not isSensory:
            data['daq'] = read_acquisition(nwbfile, downsampled=downsampled)
            for view in ('body', 'face', 'eye'):
                data[f'{view}_video_tracking'] = read_video_tracking(nwbfile, view=view, downsampled=downsampled)
            data['pupil_tracking'] = read_video_tracking(nwbfile, view='pupil', downsampled=downsampled)
        data['rois'], data['roi_description'] = read_roi_dFF(nwbfile)
    if isResting:
        return NWBData_resting(**data)
    if isSensory:
        return NWBData_sensory(**data)
    return NWBData(**data)
