# config objects
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
import pandas as pd
import cedalion.data
import pickle
import os
from dataclasses import dataclass
from typing import Callable, Any, Optional, List, Dict, Union

# import cedalion.imagereco.forward_model as fw
# from cedalion.imagereco.solver import pseudo_inverse_stacked

#############################################################################################################
# DATASET CONFIGS
#############################################################################################################

path_prefix = '/home/thomas/Dokumente/Master/Master_Thesis/HD-DOT_Classification/'
data_path_prefix = path_prefix + 'data/'
results_path_prefix = path_prefix + 'results/'
images_path_prefix = path_prefix + 'writing/images/'

fwm = cedalion.dot.forward_model.ForwardModel
# fwm = cedalion.imagereco.forward_model.ForwardModel

with open(data_path_prefix + 'parcels_colin.pickle', 'rb') as f:
    parcels_colin = pickle.load(f)

with open(data_path_prefix + 'parcels_icbm.pickle', 'rb') as f:
    parcels_icbm = pickle.load(f)

long_channels_dict = {}
probe_area_dict = {}
dataset_names = ['HD_Squeezing','BS_Laura', 'NN22_Resting_State']
data_type_names = ['HD_Squeezing', 'BS_Laura', 'Syn_Finger_Tapping'] # NN22RS gets syn finger tapping after adding synthetic hrfs
for path_suffix in dataset_names:
    with open(os.path.join(data_path_prefix, path_suffix, 'long_channels'), 'rb') as f:
        long_channels_dict[path_suffix] = pickle.load(f)
    with open(os.path.join(data_path_prefix, path_suffix, 'probe_area_data'), 'rb') as f:
        probe_area_dict[path_suffix] = pickle.load(f)


@dataclass
class DataConfig:
    
    synthetic: bool
    all_subjects: List[str]
    subjects: List[str] # subjects that don't get rejected based on SCI + PSP criterion
    n_runs: Callable
    runs: Callable
    base_path: str
    snirf_path: str
    clean_channels_path: Callable
    epochs_labels_path: Callable
    feature_slices: Dict[str, slice]
    Adot: Any  
    B: Any     
    long_channels: Any
    probe_area: Any
    sensitive_parcels: List[str]
    parcel_subset: Dict[str, List[str]]
    parcel_subset_alt: Optional[Dict[str, List[str]]] = None
    stim_template: Optional[str] = None
    subsets_path: str = 'subsets_data'
    preprocess_stim: Optional[Callable] = None
    resting_snirf_path: Optional[str] = None




def load_dataset_configs(data_types, alpha_sp=1e-3, load_sensitivity=False, test=True):

    dataset_configs = {}
    
    dOD_thresh = 0.001
    minCh = 1 
    dHbO = 10 #µM 
    dHbR = -3 #µM

    # Load matrices

    if 'Syn_Finger_Tapping' in data_types or 'Syn_Stroop' in data_types:

        # ninjanirs (resting state + synthetic)
        #Adot_ninja_colin = cedalion.datasets.get_precomputed_sensitivity('nn22_resting', head_model='icbm152')
        #Adot_ninja_colin = Adot_ninja_colin.assign_coords(parcel = ("vertex", parcels_icbm))

        #Adot_ninja_colin_ft = Adot_ninja_colin.sel(channel=channels_syn_ft)
        #parcel_dOD, parcel_mask_ninja_colin_ft = fwm.parcel_sensitivity(Adot_ninja_colin_ft, None, dOD_thresh, minCh, dHbO, dHbR)
        #sensitive_parcels_ninja_colin_ft = parcel_mask_ninja_colin_ft.where(parcel_mask_ninja_colin_ft, drop=True)["parcel"].values.tolist()

        #Adot_ninja_colin_stacked = fwm.compute_stacked_sensitivity(Adot_ninja_colin)
        #B_ninja_colin = pseudo_inverse_stacked(Adot_ninja_colin_stacked, alpha = 0.01, alpha_spatial = alpha_sp)
        #nvertices = B_ninja_colin.shape[0]//2
        #B_ninja_colin = B_ninja_colin.assign_coords({"chromo" : ("flat_vertex", ["HbO"]*nvertices  + ["HbR"]* nvertices)})
        #B_ninja_colin = B_ninja_colin.set_xindex("chromo")
        #B_ninja_colin = B_ninja_colin.assign_coords({"parcel" : ("flat_vertex", np.concatenate((Adot_ninja_colin.coords['parcel'].values, Adot_ninja_colin.coords['parcel'].values)))})

        if test:
            epo_label_path=lambda subject, run, int_scaling, spatial_scaling: f"epochs_labels/Finger_Tapping/test/sp_{spatial_scaling}/{subject}/int_{int_scaling}/run{run}_epochs_labels.pkl"
            #subjects=['sub-02']
            subjects=['sub-02', 'sub-04', 'sub-05', 'sub-06', 'sub-08', 'sub-10', 'sub-11', 'sub-13', 'sub-14', 'sub-17', 'sub-18']
        else:
            epo_label_path=lambda subject, run, int_scaling, spatial_scaling: f"epochs_labels/Finger_Tapping/new/sp_{spatial_scaling}/{subject}/int_{int_scaling}/run{run}_epochs_labels.pkl"
            subjects=['sub-02', 'sub-04', 'sub-05', 'sub-06', 'sub-08', 'sub-10', 'sub-11', 'sub-13', 'sub-14', 'sub-17', 'sub-18']

        dataset_configs['Syn_Finger_Tapping'] = DataConfig(
            synthetic=True,
            all_subjects=['sub-02', 'sub-04', 'sub-05', 'sub-06', 'sub-08', 'sub-09', 'sub-10', 'sub-11', 'sub-12', 'sub-13', 'sub-14', 'sub-16', 'sub-17', 'sub-18'],
            subjects=subjects,
            n_runs=lambda idx: 2 if idx < 5 else 1,
            runs = lambda subject: [0, 1] if subject in ['sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05', 'sub-06', 'sub-07', 'sub-08'] else [0],
            base_path=data_path_prefix + "NN22_Resting_State/",
            snirf_path= "Full_SynHRF_Data/Finger_Tapping/sp_{spatial_scale}/{subject}/nirs/{subject}_task-SynHRF_{run}_nirs.snirf",
            resting_snirf_path="NN22_RS/{subject}/nirs/{subject}_task-RS_{run}_nirs.snirf",
            clean_channels_path=lambda subject, run: f"epochs_labels/clean_channels/{subject}/{run}/clean_channels.pkl",
            epochs_labels_path=epo_label_path,
            feature_slices={"Slope": slice(0, 9), "Mean": slice(3, 10), "Max": slice(2, 8), "Min": slice(2,8)},
            Adot=None,
            B=None,
            sensitive_parcels=None,
            #parcel_subset={"SomMotA": [p for p in sensitive_parcels_ninja_colin_ft if p.startswith("SomMotA")]},
            parcel_subset=None,
            long_channels=long_channels_dict["NN22_Resting_State"],
            probe_area=probe_area_dict['NN22_Resting_State']['active_area']
        )

    if 'HD_Squeezing' in data_types or 'BS_Laura' in data_types:

        if load_sensitivity:
            with open(data_path_prefix + 'HD_Squeezing/Adot/Adot_HDSqueezing_ICBM.pickle', 'rb') as f:
                Adot_HD_Squeezing = pickle.load(f)
            #Adot_HD_Squeezing = cedalion.datasets.get_precomputed_sensitivity('fingertappingDOT', head_model='icbm152')
            Adot_HD_Squeezing = Adot_HD_Squeezing.assign_coords(parcel = ("vertex", parcels_icbm))
            parcel_dOD, parcel_mask_HD_Squeezing = fwm.parcel_sensitivity(Adot_HD_Squeezing, None, dOD_thresh, minCh, dHbO, dHbR)
            sensitive_parcels_HD_Squeezing = parcel_mask_HD_Squeezing.where(parcel_mask_HD_Squeezing, drop=True)["parcel"].values.tolist()
            Adot_HD_Squeezing_stacked = fwm.compute_stacked_sensitivity(Adot_HD_Squeezing)
            B_HD_Squeezing = pseudo_inverse_stacked(Adot_HD_Squeezing_stacked, alpha = 0.01, alpha_spatial = alpha_sp)
            nvertices = B_HD_Squeezing.shape[0]//2
            B_HD_Squeezing = B_HD_Squeezing.assign_coords({"chromo" : ("flat_vertex", ["HbO"]*nvertices  + ["HbR"]* nvertices)})
            B_HD_Squeezing = B_HD_Squeezing.set_xindex("chromo")
            B_HD_Squeezing = B_HD_Squeezing.assign_coords({"parcel" : ("flat_vertex", np.concatenate((Adot_HD_Squeezing.coords['parcel'].values, Adot_HD_Squeezing.coords['parcel'].values)))})
            parcel_subset = {"SomMotA": [p for p in sensitive_parcels_HD_Squeezing if p.startswith("SomMotA")]}
        else:
            Adot_HD_Squeezing = None
            B_HD_Squeezing = None
            sensitive_parcels_HD_Squeezing = None
            parcel_subset = None

        if test:
            epo_label_path=lambda subject, run, int_scaling, spatial_scaling: f"epochs_labels/test/{subject}/run{run}_epochs_labels.pkl"
            #subjects=['sub-173']
            subjects=['sub-170', 'sub-173', 'sub-174', 'sub-176', 'sub-177', 'sub-179', 'sub-181', 'sub-182', 'sub-183', 'sub-185']
        else:
            epo_label_path=lambda subject, run, int_scaling, spatial_scaling: f"epochs_labels/{subject}/run{run}_epochs_labels.pkl"
            subjects=['sub-170', 'sub-173', 'sub-174', 'sub-176', 'sub-177', 'sub-179', 'sub-181', 'sub-182', 'sub-183', 'sub-185']

        dataset_configs['HD_Squeezing'] = DataConfig(
            synthetic=False,
            all_subjects=['sub-170', 'sub-171', 'sub-173', 'sub-174', 'sub-176', 'sub-177', 'sub-179', 'sub-181', 'sub-182', 'sub-183', 'sub-184', 'sub-185'],
            subjects=subjects,
            n_runs=lambda _: 3,
            runs=lambda _: ['1', '2', '3'],
            preprocess_stim=lambda stim: (
                pd.concat([
                    stim[stim.trial_type == "Left"].sort_values(by="onset"),
                    stim[stim.trial_type == "Right"].sort_values(by="onset")
                ]).reset_index(drop=True).assign(duration=5.0) # Set duration to 5.0 
            ),
            base_path=data_path_prefix + "HD_Squeezing/",
            snirf_path= "HD_Ballsqueezing/{subject}/nirs/{subject}_task-BallSqueezing_run-{run}_nirs.snirf",
            clean_channels_path=lambda subject, run: f"epochs_labels/clean_channels/{subject}/{run+1}/clean_channels.pkl",
            epochs_labels_path=epo_label_path,
            feature_slices={"Slope": slice(0, 9), "Mean": slice(3, 10), "Max": slice(2, 8), "Min": slice(2,8)},
            Adot=Adot_HD_Squeezing,
            B=B_HD_Squeezing,
            sensitive_parcels=sensitive_parcels_HD_Squeezing,
            parcel_subset=parcel_subset,
            #parcel_subset=None,
            long_channels=long_channels_dict["HD_Squeezing"],
            probe_area=probe_area_dict['HD_Squeezing']['active_area']
        )
 
    if 'BS_Laura' in data_types:

        laura_sens_path = os.path.join(data_path_prefix, 'BS_Laura', "BS_Laura_YY_parcel_sens_channels")
        with open(laura_sens_path, 'rb') as f:
            channel_roi_sens_laura = pickle.load(f)

        if load_sensitivity:
            # Ballsqueezing Laura
            with open(data_path_prefix + 'BS_Laura/Adot/Adot_BSLaura_ICBM.pickle', 'rb') as f:
                Adot_BS_Laura = pickle.load(f)
            Adot_BS_Laura = Adot_BS_Laura.assign_coords(parcel = ("vertex", parcels_icbm))
            Adot_BS_Laura = Adot_BS_Laura.sel(channel=channel_roi_sens_laura)
            parcel_dOD, parcel_mask_BS_Laura = fwm.parcel_sensitivity(Adot_BS_Laura, None, dOD_thresh, minCh, dHbO, dHbR)
            sensitive_parcels_BS_Laura = parcel_mask_BS_Laura.where(parcel_mask_BS_Laura, drop=True)["parcel"].values.tolist()
            Adot_BS_Laura_stacked = fwm.compute_stacked_sensitivity(Adot_BS_Laura)
            B_BS_Laura = pseudo_inverse_stacked(Adot_BS_Laura_stacked, alpha = 0.01, alpha_spatial = alpha_sp)
            nvertices = B_BS_Laura.shape[0]//2
            B_BS_Laura = B_BS_Laura.assign_coords({"chromo" : ("flat_vertex", ["HbO"]*nvertices  + ["HbR"]* nvertices)})
            B_BS_Laura = B_BS_Laura.set_xindex("chromo")
            B_BS_Laura = B_BS_Laura.set_xindex("channel")
            B_BS_Laura = B_BS_Laura.assign_coords({"parcel" : ("flat_vertex", np.concatenate((Adot_BS_Laura.coords['parcel'].values, Adot_BS_Laura.coords['parcel'].values)))})
            parcel_subset = {"SomMotA": [p for p in sensitive_parcels_BS_Laura if p.startswith("SomMotA")]}
            parcel_subset_alt={"HD Sq Parcels": sensitive_parcels_HD_Squeezing}

        else:
            Adot_BS_Laura = None
            B_BS_Laura = None
            sensitive_parcels_BS_Laura = None
            parcel_subset = None
            parcel_subset_alt = None

        if test:
            epo_label_path=lambda subject, run, int_scaling, spatial_scaling: f"epochs_labels/test/{subject}/run{run}_epochs_labels.pkl"
            #subjects=['sub-633']
            #subjects=['sub-633', 'sub-638']
            subjects=['sub-577', 'sub-580', 'sub-586', 'sub-587', 'sub-592', 'sub-613', 'sub-618', 'sub-619', 'sub-621', 'sub-633', 'sub-638', 'sub-640']
        else:
            epo_label_path=lambda subject, run, int_scaling, spatial_scaling: f"epochs_labels/{subject}/run{run}_epochs_labels.pkl"
            #subjects=['sub-633', 'sub-638']
            subjects=['sub-577', 'sub-580', 'sub-586', 'sub-587', 'sub-592', 'sub-613', 'sub-618', 'sub-619', 'sub-621', 'sub-633', 'sub-638', 'sub-640']

        dataset_configs['BS_Laura'] = DataConfig(
            synthetic=False,
            all_subjects=['sub-547','sub-568', 'sub-577', 'sub-580', 'sub-581', 'sub-583', 'sub-586', 'sub-587', 'sub-588', 'sub-592', 'sub-613', 'sub-618', 'sub-619', 'sub-621', 'sub-633', 'sub-638', 'sub-640'],
            subjects=subjects,
            n_runs=lambda _: 3,
            runs=lambda _: ['1', '2', '3'],
            base_path=data_path_prefix + "BS_Laura/",
            snirf_path="BS_Laura_Data/{subject}/nirs/{subject}_task-BS_run-0{run}_nirs.snirf",
            stim_template = "BS_Laura_Data/{subject}/nirs/{subject}_task-BS_run-0{run}_events.tsv",
            clean_channels_path=lambda subject, run: f"epochs_labels/clean_channels/{subject}/{run+1}/clean_channels.pkl",
            epochs_labels_path=epo_label_path,
            feature_slices={"Slope": slice(0, 9), "Mean": slice(3, 10), "Max": slice(2, 8), "Min": slice(2,8)},
            Adot=Adot_BS_Laura,
            B=B_BS_Laura,
            sensitive_parcels=sensitive_parcels_HD_Squeezing,
            parcel_subset=parcel_subset,
            parcel_subset_alt=parcel_subset_alt,
            long_channels=long_channels_dict["BS_Laura"],
            probe_area=probe_area_dict['BS_Laura']['active_area']
        )

    return dataset_configs



#############################################################################################################
# RUN CONFIGS
#############################################################################################################


def sp_map(sp: int) -> str:
    return f"{sp} cm"

INT_MAP = {'01': '0.2 μM', '02': '0.4 μM', '03': '0.6 μM'}

@dataclass
class GlobalConfig:
    # classifiers & flags
    classifiers: Dict[str, Any]
    save_plot: bool = True
    run_pipeline: bool = True
    run_ch_ws: bool = True
    run_ch_loso: bool = False
    run_parcel_pipeline: bool = False
    run_parcel_ws: bool = False
    run_parcel_loso: bool = False

    # feature selection & ROI
    prune_by_zeroing_loso: bool = True
    n_reduced_feat_ws: int = None
    n_reduced_feat_loso: int = None
    sel_hrf_roi: int = None
    only_hbo: bool = True

    # data & iteration
    datasets_path: str = data_path_prefix
    result_root: str = results_path_prefix

    # common iteration axes
    int_scalings: List[str] = None
    spatial_scalings: List[int] = None

    # conditions/labels
    dt_conditions: List[str] = None
    dt_labels: Dict[str, str] = None


@dataclass
class RunContext:

    g: GlobalConfig                               
    ds: Dict[str, Any]                            
    feature_types: List[str] = None
    clf_name: str = None
    classifier: Any = None

    # toggles that vary inside loops
    reduce_features: bool = False
    prune_channels: bool = True
    prune_chans_sma: bool = False

    # loop indices/context
    data_type: str = ""
    int_scaling: str = "01"
    spatial_scaling: int = 1
    dt: str = None
    sub_key: str = None
    subject: Optional[str] = None
    run: Optional[int] = None

    # loaded/derived
    subsets_data: Optional[Dict[str, Dict[str, Any]]] = None
    clean_ch_map: Optional[Dict[str, Dict[int, Dict[str, List[str]]]]] = None
    sma_sens_channels: Optional[List[str]] = None

    ft_slices: Dict[str, slice] = field(default_factory=dict)


syn_cfg = GlobalConfig(
    classifiers={'Linear_SVM': SVC(kernel='linear', C=0.1, max_iter=10000)},
    #classifiers={'LDA': LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')},
    #classifiers={'LDA': LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'),
    #             'Linear_SVM': SVC(kernel='linear', C=0.1, max_iter=10000)},
    save_plot=True, int_scalings=['01', '02', '03'], spatial_scalings=[1, 2, 3],
    run_pipeline=True, run_ch_ws=True, run_ch_loso=False,
    run_parcel_pipeline=False, run_parcel_ws=False, run_parcel_loso=False,
    prune_by_zeroing_loso=True, sel_hrf_roi=20, only_hbo=True,
    dt_conditions=["all", "all_ss_mean"],
    dt_labels={"all": "No SS Correction", "all_ss_mean": "SS corrected"},
)

hd_sq_cfg = GlobalConfig(
    classifiers={'Linear_SVM': SVC(kernel='linear', C=0.1, max_iter=10000)},
    #classifiers={'LDA': LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')},
    #classifiers={'LDA': LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'),
    #             'Linear_SVM': SVC(kernel='linear', C=0.1, max_iter=10000)},
    save_plot=True, int_scalings=['01'], spatial_scalings=[1],
    run_pipeline=True, run_ch_ws=True, run_ch_loso=True,
    run_parcel_pipeline=False, run_parcel_ws=True, run_parcel_loso=True,
    prune_by_zeroing_loso=True, sel_hrf_roi=None, only_hbo=True,
    dt_conditions=["all", "all_ss_mean"],
    dt_labels={"all": "No SS Correction", "all_ss_mean": "SS corrected"},
)

bsl_cfg = GlobalConfig(
    classifiers={'Linear_SVM': SVC(kernel='linear', C=0.1, max_iter=10000)},
    #classifiers={'LDA': LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')},
    #classifiers={'LDA': LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'),
    #             'Linear_SVM': SVC(kernel='linear', C=0.1, max_iter=10000)},
    save_plot=True, int_scalings=['01'], spatial_scalings=[1],
    run_pipeline=True, run_ch_ws=True, run_ch_loso=True,
    run_parcel_pipeline=False, run_parcel_ws=True, run_parcel_loso=True,
    prune_by_zeroing_loso=True, sel_hrf_roi=20, only_hbo=True,
    dt_conditions=["all", "all_ss_mean"],
    dt_labels={"all": "No SS Correction", "all_ss_mean": "SS corrected"},
)
 