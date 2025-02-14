import os
import re
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import bdbc_nwb_tools as nwbtools

file_matcher = re.compile(r'([a-zA-Z0-9-]+)_([0-9-]+)_(task|resting-state|sensory-stim)-day([0-9]+)')
type_indexer = ['task', 'resting-state', 'sensory-stim']

def sanity_check_nwb(nwb_path,
                     output_txt_path = None,
                     isResting = False,
                     isSensory = False
                     ) -> None:
    
    # load the nwb file and print summary
    assert os.path.exists(nwb_path), f'File not found: {nwb_path}'

    if output_txt_path is None:
        output_txt_path = nwb_path.replace('.nwb', '_summary.txt')

    session = nwbtools.load_from_file(nwb_path,
                                      downsampled=True,
                                      isResting=isResting,
                                      isSensory=isSensory)
 
    if isSensory:
        df_name_list = ['rois',]
    else: 
        df_name_list = ['rois',
                        'daq',
                        'body_video_tracking',
                        'face_video_tracking',
                        'eye_video_tracking',
                        'pupil_tracking']
    
    with open(output_txt_path, 'w') as f:
        for df_name in df_name_list:
            summary = stat_summary(session, df_name, nwb_path)
            f.write(f"\n\n\n{df_name.upper()} summary\n")
            f.write(summary.to_string(index=True, header = True))

def stat_summary(session:nwbtools.NWBData, df_name, nwb_path, verbose = True):
    tab = getattr(session, df_name)
    if tab is None: 
        if verbose:
            print(f"***no {df_name} found in {nwb_path}")
            print()
        return None
    summary = tab.describe().T
    summary = summary[['mean', 'std', 'min', 'max']]
    summary['nan_count'] = tab.isna().sum() 
    if verbose:
        print(f"{df_name} summary for {nwb_path}")
        print(summary)
        print()
    return summary


def nwb_file_indexer(filepath: Path) -> tuple[str, str, str, int]:
    matches = file_matcher.match(filepath.stem)
    if not matches:
        raise ValueError(f"file does not match: {filepath.stem}")
    return (matches.group(1), matches.group(2), type_indexer.index(matches.group(3)), int(matches.group(4)))


def plot_summary_within_animal(folder_path, output_folder = None):
    assert os.path.exists(folder_path)
    folder_name = str(folder_path).split(os.path.sep)[-1]  # animal name

    # find all nwb files in the folder
    nwb_files = sorted(Path(folder_path).rglob("*.nwb"), key=nwb_file_indexer)
    isRestings = ['resting' in nwb_file.name for nwb_file in nwb_files]
    isSensorys = ['sensory' in nwb_file.name for nwb_file in nwb_files]

    # get list of session
    sessions = []
    for nwb_file, isResting, isSensory in zip(nwb_files, isRestings, isSensorys):
        print(f"loading {nwb_file}")
        sessions.append(nwbtools.load_from_file(nwb_file,
                                                downsampled = True,
                                                isResting = isResting,
                                                isSensory = isSensory
                                                ))

    df_name_list = [
                            'daq',
                            'body_video_tracking',
                            'face_video_tracking',
                            'eye_video_tracking',
                            'pupil_tracking', 'rois',]


    for df_name in df_name_list:
        if output_folder is None:
            png_path_base = os.path.join(str(folder_path), 'summary', f'{folder_name}_{df_name}')
        else:
            png_path_base = os.path.join(str(output_folder), f'{folder_name}_{df_name}')

        file_names = [nwb_file.name for nwb_file in nwb_files]

        if df_name == 'rois':
            plot_stats_across_sessions(file_names, sessions, png_path_base = png_path_base, df_name = df_name)

        else: # sensory data should be omitted
            nwb_files_no_sensory = [nwb_file for nwb_file, isSensory in zip(file_names, isSensorys) if not isSensory]
            sessions_no_sensory = [session for session, isSensory in zip(sessions, isSensorys) if not isSensory]
            plot_stats_across_sessions(nwb_files_no_sensory, sessions_no_sensory, png_path_base = png_path_base, df_name = df_name)


def plot_stats_across_sessions(nwb_files, sessions, png_path_base = None, df_name = 'rois'):

    summarys = []  # each item having shape (num_features, num_stats)
    rows = None  # corresponds to the types of the features to be plotted
    for i in range(len(nwb_files)):
        summary = stat_summary(sessions[i], df_name, None, verbose=False)
        if summary is not None:
            summarys.append(summary)
            if rows is None:
                rows = summary.index

    nRow = int(len(rows)/2) if df_name == 'rois' else len(rows)

    maxRow = 10
    nCol = 5 # mean, max, min, std, nan

    stats = ['mean', 'max', 'min', 'std', 'nan_count']

    fig = plt.figure(figsize=(12, 15), facecolor='w')
    gs = gridspec.GridSpec(maxRow+2, nCol, figure=fig)

    for i in range(nRow):
        iRow = i % maxRow
        if df_name == 'rois':
            # left 
            row_name = rows[i][:-2] 

            selected_row_l = [df.loc[f"{row_name}_l"] for df in summarys]  # does not have to take None's into account (b/c it is from imaging data)
            result_l = pd.concat(selected_row_l, axis=0)

            # right
            selected_row_r = [df.loc[f"{row_name}_r"] for df in summarys]  # does not have to take None's into account (b/c it is from imaging data)
            result_r = pd.concat(selected_row_r, axis=0)

            for j, stat in enumerate(stats):

                ax = fig.add_subplot(gs[iRow, j])

                ax.plot(result_l[stat].values, 'ro-', label=stat)
                ax.plot(result_r[stat].values, 'bo-', label=stat)
                if j == 0:
                    ax.set_ylabel(row_name)
                if iRow == 0:
                    ax.set_title(stat)

        else:
            row_name = rows[i]
            selected_row = []
            rowidxx = []
            for rowidx, df in enumerate(summarys, start=1):
                if df is not None:
                    selected_row.append(df.iloc[i])
                    rowidxx.append(rowidx)
            result = pd.concat(selected_row, axis=0)

            for j, stat in enumerate(stats):
                ax = fig.add_subplot(gs[iRow, j])
                ax.plot(rowidxx, result[stat].values, 'ro-', label=stat)
                if j == 0:
                    ax.set_ylabel(row_name)
                if iRow == 0:
                    ax.set_title(stat)
            
        if i % maxRow == maxRow-1 or i == nRow-1:
            #add text
            textlist = [f"{i} : {j}" for i, j in enumerate(nwb_files)]
            numline = len(textlist)//2 + 1
            for j in range(2):
                text_part = "\n".join(textlist[j*numline:(j+1)*numline])
                ax_text = fig.add_subplot(gs[-2:, j*2:(j+1)*2])
                ax_text.axis('off')
                ax_text.text(0.05, 0.95, text_part, transform=ax_text.transAxes, fontsize=11,
                        verticalalignment='top', bbox=dict(facecolor='w', alpha=0.5),
                        )


            plt.tight_layout()
            png_path = f"{png_path_base}_{(i // maxRow) + 1:02d}.png"

            # create folder if not exist
            if not os.path.exists(os.path.dirname(png_path)):
                os.makedirs(os.path.dirname(png_path))

            fig.savefig(png_path, dpi=300)
            plt.close(fig)
            print(f"saved {png_path}")

            # create new figure again
            fig = plt.figure(figsize=(12, 15), facecolor='w')
            gs = gridspec.GridSpec(maxRow+2, 5, figure=fig)
    plt.close(fig)


# test code
if __name__ == '__main__':

    root_path = r"path-to-root-folder"
    assert os.path.exists(root_path)

    # find all folders in root
    folders = [d for d in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, d))]
    print(folders)

    for folder in folders:
        print(f"processing {folder}...")
        folder_path = os.path.join(root_path, folder)
        plot_summary_within_animal(folder_path)
