""" top level run script """

import glob
import os
import shutil

import numpy as np
from scipy import signal

from pathlib import Path
from tqdm import tqdm

import spikeinterface as si
import spikeinterface.extractors as se
import spikeinterface.exporters as sexp

import one.alf.io as alfio

from utils import WindowGenerator, fscale, hp, rms

data_folder = Path('../data')
scratch_folder = Path('../scratch')
results_folder = Path('../results')

# define parameters
RMS_WIN_LENGTH_SECS = 3
WELCH_WIN_LENGTH_SAMPLES = 1024
TOTAL_SECS = 100

job_kwargs = dict(n_jobs=-1)


if __name__ == "__main__":

    si.set_global_job_kwargs(**job_kwargs)

    ecephys_sessions = [p for p in data_folder.iterdir() if "ecephys" in p.name.lower() and "sorted" not in p.name.lower()]
    assert len(ecephys_sessions) == 1, f"Attach one session at a time {ecephys_sessions}"
    session_folder = ecephys_sessions[0]
    ecephys_folder = session_folder / "ecephys_clipped"
    ecephys_compressed_folder = session_folder / 'ecephys_compressed'

    # check if capsule mode
    pipeline_mode = True
    ecephys_processed_folders = [p for p in data_folder.iterdir() if "sorted" in p.name.lower() and session_folder.name in p.name.lower()]
    if len(ecephys_processed_folders) == 1:
        pipeline_mode = False
        ecephys_processed_folder = ecephys_processed_folders[0]
    elif len(ecephys_processed_folders) > 1:
        raise Exception

    if pipeline_mode:
        # check if test
        if (data_folder / "postprocessing_pipeline_output_test").is_dir():
            print("\n*******************\n**** TEST MODE ****\n*******************\n")
            postprocessed_folder = data_folder / "postprocessing_pipeline_output_test"
        else:
            postprocessed_folder = data_folder
    else:
        postprocessed_folder = ecephys_processed_folder / "postprocessed"
    # get recording_names from postprocessing
    recording_names = [p.name for p in postprocessed_folder.iterdir() if p.is_dir() and "-sorting" not in p.name]

    for recording_name in recording_names:
        print(f"\tConverting continuous {recording_name}")

        # Find stream name: "experiment*_{stream_name}_recording*"
        stream_name = "_".join(recording_name.split("_")[1:-1])

        output_folder = results_folder / recording_name

        if not output_folder.is_dir():
            output_folder.mkdir()

        # check for LF
        lf_stream_name = stream_name.replace("AP", "LFP")

        if '-LFP' in stream_name:
            is_lfp = True
            np2 = False
            ap_stream_name = stream_name.replace("LFP", "AP")
        elif '-AP' in stream_name:
            is_lfp = False
            ap_stream_name = stream_name
        else: # Neuropixels 2.0
            is_lfp = True
            ap_stream_name = stream_name
            
        waveform_folder = postprocessed_folder / recording_name            
        we_recless = si.load_waveforms(
            waveform_folder, 
            with_recording=False
        )

        recording = si.read_zarr(ecephys_compressed_folder / f"experiment1_{stream_name}.zarr")
        recording_lfp_folder = ecephys_compressed_folder / f"experiment1_{lf_stream_name}.zarr"
        if recording_lfp_folder.is_dir():
            recording_lfp = si.read_zarr(recording_lfp_folder)
        else: #NP2
            # filter and downsample?
            recording_lfp = None

        channel_inds, = np.isin(recording.channel_ids, we_recless.channel_ids).nonzero()

        fs_ap = recording.sampling_frequency
        rms_win_length_samples_ap = 2 ** np.ceil(np.log2(fs_ap * RMS_WIN_LENGTH_SECS))
        total_samples_ap = int(np.min([fs_ap * TOTAL_SECS, recording.get_num_samples()]))

        # the window generator will generates window indices
        wingen = WindowGenerator(ns=total_samples_ap, nswin=rms_win_length_samples_ap, overlap=0)
        win = {
            'TRMS': np.zeros((wingen.nwin, recording.get_num_channels())),
            'nsamples': np.zeros((wingen.nwin,)),
            'fscale': fscale(WELCH_WIN_LENGTH_SAMPLES, 1 / fs_ap, one_sided=True),
            'tscale': wingen.tscale(fs=fs_ap)
        }
        win['spectral_density'] = np.zeros((len(win['fscale']), recording.get_num_channels()))

        # @Josh: this could be dramatically sped up if we employ SpikeInterface parallelization
        with tqdm(total=wingen.nwin) as pbar:
            for first, last in wingen.firstlast:
                D = recording.get_traces(start_frame=first, end_frame=last).T
                # remove low frequency noise below 1 Hz
                D = hp(D, 1 / fs_ap, [0, 1])
                iw = wingen.iw
                win['TRMS'][iw, :] = rms(D)
                win['nsamples'][iw] = D.shape[1]
                
                # the last window may be smaller than what is needed for welch
                if last - first < WELCH_WIN_LENGTH_SAMPLES:
                    continue
                
                # compute a smoothed spectrum using welch method
                _, w = signal.welch(
                    D, fs=fs_ap, window='hann', nperseg=WELCH_WIN_LENGTH_SAMPLES,
                    detrend='constant', return_onesided=True, scaling='density', axis=-1
                )
                win['spectral_density'] += w.T
                # print at least every 20 windows
                if (iw % min(20, max(int(np.floor(wingen.nwin / 75)), 1))) == 0:
                    pbar.update(iw)
                    
        win['TRMS'] = win['TRMS'][:,channel_inds]
        win['spectral_density'] = win['spectral_density'][:,channel_inds]
    
        alf_object_time = f'ephysTimeRmsAP'
        alf_object_freq = f'ephysSpectralDensityAP'

        tdict = {
            'rms': win['TRMS'].astype(np.single), 
            'timestamps': win['tscale'].astype(np.single)
        }
        alfio.save_object_npy(output_folder, object=alf_object_time, dico=tdict, namespace='iblqc')
        
        fdict = {
            'power': win['spectral_density'].astype(np.single),
            'freqs': win['fscale'].astype(np.single)
        }
        alfio.save_object_npy(
            output_folder, object=alf_object_freq, dico=fdict, namespace='iblqc'
        )

        if recording_lfp is not None:
            print("\tConverting LFP stream")
            fs_lfp = recording_lfp.sampling_frequency
            rms_win_length_samples_lfp = 2 ** np.ceil(np.log2(fs_lfp * RMS_WIN_LENGTH_SECS))
            total_samples_lfp = int(np.min([fs_lfp * TOTAL_SECS, recording_lfp.get_num_samples()]))

            # the window generator will generates window indices
            wingen = WindowGenerator(ns=total_samples_lfp, nswin=rms_win_length_samples_lfp, overlap=0)
            win = {
                'TRMS': np.zeros((wingen.nwin, recording_lfp.get_num_channels())),
                'nsamples': np.zeros((wingen.nwin,)),
                'fscale': fscale(WELCH_WIN_LENGTH_SAMPLES, 1 / fs_lfp, one_sided=True),
                'tscale': wingen.tscale(fs=fs_lfp)
            }
            win['spectral_density'] = np.zeros((len(win['fscale']), recording_lfp.get_num_channels()))

            # @Josh: this could be dramatically sped up if we employ SpikeInterface parallelization
            with tqdm(total=wingen.nwin) as pbar:
                for first, last in wingen.firstlast:
                    D = recording_lfp.get_traces(start_frame=first, end_frame=last).T
                    # remove low frequency noise below 1 Hz
                    D = hp(D, 1 / recording.sampling_frequency, [0, 1])
                    iw = wingen.iw
                    win['TRMS'][iw, :] = rms(D)
                    win['nsamples'][iw] = D.shape[1]
                    
                    # the last window may be smaller than what is needed for welch
                    if last - first < WELCH_WIN_LENGTH_SAMPLES:
                        continue
                    
                    # compute a smoothed spectrum using welch method
                    _, w = signal.welch(
                        D, fs=fs_lfp, window='hann', nperseg=WELCH_WIN_LENGTH_SAMPLES,
                        detrend='constant', return_onesided=True, scaling='density', axis=-1
                    )
                    win['spectral_density'] += w.T
                    # print at least every 20 windows
                    if (iw % min(20, max(int(np.floor(wingen.nwin / 75)), 1))) == 0:
                        pbar.update(iw)
                        
            win['TRMS'] = win['TRMS'][:,channel_inds]
            win['spectral_density'] = win['spectral_density'][:,channel_inds]
        
            alf_object_time = f'ephysTimeRmsLF'
            alf_object_freq = f'ephysSpectralDensityLF'

            tdict = {
                'rms': win['TRMS'].astype(np.single), 
                'timestamps': win['tscale'].astype(np.single)
            }
            alfio.save_object_npy(output_folder, object=alf_object_time, dico=tdict, namespace='iblqc')
            
            fdict = {
                'power': win['spectral_density'].astype(np.single),
                'freqs': win['fscale'].astype(np.single)
            }
            alfio.save_object_npy(
                output_folder, object=alf_object_freq, dico=fdict, namespace='iblqc'
            )

        print(f"\n\tConverting spikes")
        phy_folder = scratch_folder / f"{recording_name}_phy"
    
        print('\t\tExporting to phy format...')
        sexp.export_to_phy(
            we_recless, 
            output_folder=phy_folder,
            compute_pc_features=False,
            remove_if_exists=True,
            copy_binary=False,
            verbose=False
        )

        spike_locations = we_recless.load_extension("spike_locations").get_data()
        spike_depths = spike_locations["y"]

        print('\t\tConverting data...')
        # convert clusters and squeeze
        clusters = np.load(phy_folder / "spike_clusters.npy")
        np.save(phy_folder / "spike_clusters.npy",
                np.squeeze(clusters.astype('uint32')))
        
        # convert times and squeeze
        times = np.load(phy_folder / "spike_times.npy")
        np.save(phy_folder / "spike_times.npy", np.squeeze(times / 30000.).astype('float64'))
        
        # convert amplitudes and squeeze
        amps = np.load(phy_folder / "amplitudes.npy")
        np.save(phy_folder / "amplitudes.npy", np.squeeze(-amps / 1e6).astype('float64'))
        
        # save depths and channel inds
        np.save(phy_folder / "spike_depths.npy", spike_depths)
        np.save(phy_folder / "channel_inds.npy", np.arange(len(channel_inds), dtype='int'))
        
        # save templates
        cluster_channels = []
        cluster_peakToTrough = []
        cluster_waveforms = []
        num_chans = []

        templates = we_recless.get_all_templates()
        channel_locs = we_recless.get_channel_locations()
        extremum_channel_indices = si.get_template_extremum_channel(we_recless, outputs="index")

        for unit_idx, unit_id in enumerate(we_recless.unit_ids):
            waveform = templates[unit_idx, :, :]
            extremum_channel_index = extremum_channel_indices[unit_id]
            peak_waveform = waveform[:, extremum_channel_index]
            peakToTrough = (np.argmax(peak_waveform) - np.argmin(peak_waveform)) / we_recless.sampling_frequency
            cluster_channels.append(int(channel_locs[extremum_channel_index, 1] / 10)) #???
            cluster_peakToTrough.append(peakToTrough)
            cluster_waveforms.append(waveform)

        np.save(phy_folder / "cluster_peakToTrough.npy", np.array(cluster_peakToTrough))
        np.save(phy_folder / "cluster_waveforms.npy", np.stack(cluster_waveforms))
        np.save(phy_folder / "cluster_channels.npy", np.array(cluster_channels))

        # rename files
        _FILE_RENAMES = [  # file_in, file_out
                ('channel_positions.npy', 'channels.localCoordinates.npy'),
                ('channel_inds.npy', 'channels.rawInd.npy'),
                ('cluster_peakToTrough.npy', 'clusters.peakToTrough.npy'),
                ('cluster_channels.npy', 'clusters.channels.npy'),
                ('cluster_waveforms.npy', 'clusters.waveforms.npy'),
                ('spike_clusters.npy', 'spikes.clusters.npy'),
                ('amplitudes.npy', 'spikes.amps.npy'),
                ('spike_depths.npy', 'spikes.depths.npy'),
                ('spike_times.npy', 'spikes.times.npy'),
            ]


        for names in _FILE_RENAMES:
            old_name = phy_folder / names[0]
            new_name = output_folder / names[1]
            shutil.copyfile(old_name, new_name)

        # save quality metrics
        qm = we_recless.load_extension("quality_metrics")
        qm_data = qm.get_data()
        
        qm_data.index.name = 'cluster_id'
        qm_data['cluster_id.1'] = qm_data.index.values
        
        qm_data.to_csv(output_folder / 'clusters.metrics.csv')
