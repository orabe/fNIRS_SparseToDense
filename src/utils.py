import pandas as pd
import numpy as np
import os
import glob
import random

def create_train_test_files(bids_path, preprocessed_path, test_subjects_list=None, test_subject_percentage=0.2):
    """
    Create train and test files for the dataset.
    """
    participants_df = pd.read_csv(os.path.join(bids_path, "participants.tsv"), sep="\t")

    if test_subjects_list is not None:
        for test_subject in test_subjects_list:
            if test_subject not in participants_df["participant_id"].values:
                raise ValueError(f"Test subject {test_subject} not found in participants.tsv")
    else:
        # Randomly select test subjects
        num_test_subjects = int(len(participants_df) * test_subject_percentage)
        test_subjects = participants_df.sample(n=num_test_subjects, random_state=42)["participant_id"].values
        test_subjects_list = list(test_subjects)

    train_subjects_list = [sub for sub in participants_df["participant_id"].values if sub not in test_subjects_list]
    print(f"Number of train subjects: {len(train_subjects_list)}")


    test_df = pd.DataFrame()
    train_df = pd.DataFrame()

    events_files = glob.glob(os.path.join(bids_path, "**", "*_events.tsv"), recursive=True)
    for events_file in events_files:
        snirf_file = events_file.replace(bids_path, preprocessed_path).replace("_events.tsv", "_nirs.nc")
        if not os.path.exists(snirf_file):
            print(f"SNIRF file not found for {snirf_file}. Skipping...")
            continue
        # Read the events file
        events_df = pd.read_csv(events_file, sep="\t")
        # Get the subject ID from the filename
        subject_id = os.path.basename(events_file).split("_")[0]
        events_df["subject_id"] = subject_id
        events_df = events_df.drop(columns=["value"])
        events_df["snirf_file"] = snirf_file
        events_df['trial_type'] = events_df['trial_type'].map({"Left": 0, "Right": 1})
        # Check if the subject is in the test list
        if subject_id in test_subjects_list:
            # Save the test events file
            test_df = pd.concat([test_df, events_df])

        else:
            # Save the train events file
            train_df = pd.concat([train_df, events_df])

    # Save the test and train DataFrames to CSV files
    test_df.to_csv(os.path.join(bids_path, "test_events.csv"), index=False)
    train_df.to_csv(os.path.join(bids_path, "train_events.csv"), index=False)
    return os.path.join(bids_path, "train_events.csv"), os.path.join(bids_path, "test_events.csv")

def create_train_test_segments_aug(bids_path, preprocessed_path, test_subjects_list=None, test_subject_percentage=0.2, exclude_subjects=None):
    """
    Create train and test files for the dataset.
    """
    if bids_path is not None:
        participants_df = pd.read_csv(os.path.join(bids_path, "participants.tsv"), sep="\t")
    else:
        participants = glob.glob(preprocessed_path + "/sub-*")
        participants = [os.path.basename(p) for p in participants]
        participants_df = pd.DataFrame({"participant_id": participants})

    if exclude_subjects is not None:
        participants_df = participants_df[~participants_df["participant_id"].isin(exclude_subjects)]
        print(f"Excluding subjects: {exclude_subjects}")

    if test_subjects_list is not None:
        for test_subject in test_subjects_list:
            if test_subject not in participants_df["participant_id"].values:
                raise ValueError(f"Test subject {test_subject} not found in participants.tsv")
    else:
        # Randomly select test subjects
        num_test_subjects = int(len(participants_df) * test_subject_percentage)
        test_subjects = participants_df.sample(n=num_test_subjects, random_state=42)["participant_id"].values
        test_subjects_list = list(test_subjects)

    train_subjects_list = [sub for sub in participants_df["participant_id"].values if sub not in test_subjects_list]
    print(f"Number of train subjects: {len(train_subjects_list)}")


    test_df = pd.DataFrame()
    train_df = pd.DataFrame()

    labels = {"Left": 0, "Right": 1, "left": 0, "right": 1}

    train_files =  []
    for train_subject in train_subjects_list:
        train_files += glob.glob(os.path.join(preprocessed_path, train_subject, "**", "*.nc"))
    train_labels = []
    for f in train_files:
        if os.path.basename(f).endswith("_test.nc"):
            train_labels.append(labels[os.path.basename(f).split("_")[-3]])
        else:
            train_labels.append(labels[os.path.basename(f).split("_")[-2]])
    train_df = pd.DataFrame({
        "snirf_file": train_files,
        "trial_type": train_labels})
    
    test_files =  []
    for test_subject in test_subjects_list:
        test_files += glob.glob(os.path.join(preprocessed_path, test_subject, "**", "*_test.nc"))
    test_labels = []
    for f in test_files:
        if os.path.basename(f).endswith("_test.nc"):
            test_labels.append(labels[os.path.basename(f).split("_")[-3]])
        else:
            test_labels.append(labels[os.path.basename(f).split("_")[-2]])
    test_df = pd.DataFrame({
        "snirf_file": test_files,
        "trial_type": test_labels})

    # Save the test and train DataFrames to CSV files
    test_df.to_csv(os.path.join(preprocessed_path, "test_segments.csv"), index=False)
    train_df.to_csv(os.path.join(preprocessed_path, "train_segments.csv"), index=False)
    return os.path.join(preprocessed_path, "train_segments.csv"), os.path.join(preprocessed_path, "test_segments.csv")

def create_train_test_segments(bids_path, preprocessed_path, test_subjects_list=None, test_subject_percentage=0.2, exclude_subjects=None):
    """
    Create train and test files for the dataset.
    """
    # Load the participants.tsv file
    if bids_path is not None:
        participants_df = pd.read_csv(os.path.join(bids_path, "participants.tsv"), sep="\t")
    else:
        participants = glob.glob(preprocessed_path + "/sub-*")
        participants = [os.path.basename(p) for p in participants]
        participants_df = pd.DataFrame({"participant_id": participants})

    if exclude_subjects is not None:
        participants_df = participants_df[~participants_df["participant_id"].isin(exclude_subjects)]
        print(f"Excluding subjects: {exclude_subjects}")

    if test_subjects_list is not None:
        for test_subject in test_subjects_list:
            if test_subject not in participants_df["participant_id"].values:
                raise ValueError(f"Test subject {test_subject} not found in participants.tsv")
    else:
        # Randomly select test subjects
        num_test_subjects = int(len(participants_df) * test_subject_percentage)
        test_subjects = participants_df.sample(n=num_test_subjects, random_state=42)["participant_id"].values
        test_subjects_list = list(test_subjects)

    train_subjects_list = [sub for sub in participants_df["participant_id"].values if sub not in test_subjects_list]
    print(f"Number of train subjects: {len(train_subjects_list)}")


    test_df = pd.DataFrame()
    train_df = pd.DataFrame()

    labels = {"Left": 1, "Right": 0, "left": 1, "right": 0}

    train_files =  []
    for train_subject in train_subjects_list:
        # train_files += glob.glob(os.path.join(preprocessed_path, train_subject, "**", "*.nc"))
        train_files += glob.glob(os.path.join(preprocessed_path, train_subject, "**", "*.nc"), recursive=True)
    train_labels = []
    for f in train_files:
        if os.path.basename(f).endswith("_test.nc"):
            train_labels.append(labels[os.path.basename(f).split("_")[-3]])
        else:
            train_labels.append(labels[os.path.basename(f).split("_")[-2]])
    train_df = pd.DataFrame({
        "snirf_file": train_files,
        "trial_type": train_labels})
    
    test_files =  []
    for test_subject in test_subjects_list:
        # test_files += glob.glob(os.path.join(preprocessed_path, test_subject, "**", "*_test.nc"))
        test_files += glob.glob(os.path.join(preprocessed_path, test_subject, "**", "*_test.nc"), recursive=True)
    test_labels = []
    for f in test_files:
        if os.path.basename(f).endswith("_test.nc"):
            test_labels.append(labels[os.path.basename(f).split("_")[-3]])
        else:
            test_labels.append(labels[os.path.basename(f).split("_")[-2]])
    test_df = pd.DataFrame({
        "snirf_file": test_files,
        "trial_type": test_labels})

    # Save the test and train DataFrames to CSV files
    test_df.to_csv(os.path.join(preprocessed_path, "test_segments.csv"), index=False)
    train_df.to_csv(os.path.join(preprocessed_path, "train_segments.csv"), index=False)
    return os.path.join(preprocessed_path, "train_segments.csv"), os.path.join(preprocessed_path, "test_segments.csv")

def create_train_test_segments_grad(bids_path, preprocessed_path, test_subjects_list=None, exclude_subjects=None, train_dataset=None, fmri_percentage=0.2, fmri_subjects=24):
    """
    Create train and test files for the dataset.
    """
    # Load the participants.tsv file
    # yuanyuan = ["sub-170", "sub-171", "sub-173", "sub-174", "sub-176", "sub-177", 
    # "sub-179", "sub-181", "sub-182", "sub-183", "sub-184", "sub-185"]
    # laura = ["sub-547", "sub-568", "sub-577", "sub-580", "sub-581", "sub-583", "sub-586",
    # "sub-587", "sub-588", "sub-592", "sub-613", "sub-618", "sub-619", "sub-621", "sub-633",
    # "sub-638", "sub-639", "sub-640"]
    yuanyuan = ['sub-177', 'sub-182', 'sub-185', 'sub-633', 'sub-176', 'sub-580', 'sub-583', 'sub-586', 'sub-618', 'sub-640', 'sub-568', 'sub-621']
    laura = ['sub-179', 'sub-183', 'sub-581', 'sub-181', 'sub-587', 'sub-577', 'sub-638', 'sub-619', 'sub-613', 'sub-592', 'sub-170', 'sub-173']

    if bids_path is not None:
        participants_df = pd.read_csv(os.path.join(bids_path, "participants.tsv"), sep="\t")
    else:
        participants = glob.glob(preprocessed_path + "/sub-*")
        participants = [os.path.basename(p) for p in participants]
        participants_df = pd.DataFrame({"participant_id": participants})

    if exclude_subjects is not None:
        participants_df = participants_df[~participants_df["participant_id"].isin(exclude_subjects)]
        print(f"Excluding subjects: {exclude_subjects}")

    if test_subjects_list is not None:
        for test_subject in test_subjects_list:
            if test_subject not in participants_df["participant_id"].values:
                raise ValueError(f"Test subject {test_subject} not found in participants.tsv")
    else:
        # Randomly select test subjects
        num_test_subjects = int(len(participants_df) * test_subject_percentage)
        test_subjects = participants_df.sample(n=num_test_subjects, random_state=42)["participant_id"].values
        test_subjects_list = list(test_subjects)

    if train_dataset == "yuanyuan":
        data_list = yuanyuan
    elif train_dataset == "laura":
        data_list = laura + yuanyuan
    elif train_dataset == "fmri":
        data_list = participants_df["participant_id"].values
        fmri = [sub for sub in data_list if sub not in laura + yuanyuan]
        fmri_data = random.sample(fmri, int(fmri_subjects))
        data_list = laura + yuanyuan + fmri_data
    else:
        data_list = participants_df["participant_id"].values
        fmri = [sub for sub in data_list if sub not in yuanyuan]
        fmri_data = random.sample(fmri, int(fmri_subjects))
        data_list = yuanyuan + fmri_data        

    train_subjects_list = [sub for sub in participants_df["participant_id"].values if sub not in test_subjects_list and sub in data_list]
    print(f"Number of train subjects: {len(train_subjects_list)}")


    test_df = pd.DataFrame()
    train_df = pd.DataFrame()

    labels = {"Left": 0, "Right": 1, "left": 0, "right": 1}

    train_files =  []
    for train_subject in train_subjects_list:
        train_files += glob.glob(os.path.join(preprocessed_path, train_subject, "**", "*_test.nc"))
    train_labels = []
    for f in train_files:
        if os.path.basename(f).endswith("_test.nc"):
            train_labels.append(labels[os.path.basename(f).split("_")[-3]])
        else:
            train_labels.append(labels[os.path.basename(f).split("_")[-2]])
    train_df = pd.DataFrame({
        "snirf_file": train_files,
        "trial_type": train_labels})
    
    test_files =  []
    for test_subject in test_subjects_list:
        # test_files += glob.glob(os.path.join(preprocessed_path, test_subject, "**", "*_test.nc"))
        test_files += glob.glob(os.path.join(preprocessed_path, test_subject, "**", "*_test.nc"), recursive=True)
    test_labels = []
    for f in test_files:
        if os.path.basename(f).endswith("_test.nc"):
            test_labels.append(labels[os.path.basename(f).split("_")[-3]])
        else:
            test_labels.append(labels[os.path.basename(f).split("_")[-2]])
    test_df = pd.DataFrame({
        "snirf_file": test_files,
        "trial_type": test_labels})

    # Save the test and train DataFrames to CSV files
    test_df.to_csv(os.path.join(preprocessed_path, "test_segments.csv"), index=False)
    train_df.to_csv(os.path.join(preprocessed_path, "train_segments.csv"), index=False)
    return os.path.join(preprocessed_path, "train_segments.csv"), os.path.join(preprocessed_path, "test_segments.csv")


def create_train_test_segments_wustl(bids_path, preprocessed_path, test_subjects_list=None, test_subject_percentage=0.2, exclude_subjects=None):
    """
    Create train and test files for the dataset.
    """
    # Load the participants.tsv file
    if bids_path is not None:
        participants_df = pd.read_csv(os.path.join(bids_path, "participants.tsv"), sep="\t")
    else:
        participants = glob.glob(preprocessed_path + "/sub-*")
        participants = [os.path.basename(p) for p in participants]
        participants_df = pd.DataFrame({"participant_id": participants})

    if exclude_subjects is not None:
        participants_df = participants_df[~participants_df["participant_id"].isin(exclude_subjects)]
        print(f"Excluding subjects: {exclude_subjects}")

    if test_subjects_list is not None:
        for test_subject in test_subjects_list:
            if test_subject not in participants_df["participant_id"].values:
                raise ValueError(f"Test subject {test_subject} not found in participants.tsv")
    else:
        # Randomly select test subjects
        num_test_subjects = int(len(participants_df) * test_subject_percentage)
        test_subjects = participants_df.sample(n=num_test_subjects, random_state=42)["participant_id"].values
        test_subjects_list = list(test_subjects)

    train_subjects_list = [sub for sub in participants_df["participant_id"].values if sub not in test_subjects_list]
    print(f"Number of train subjects: {len(train_subjects_list)}")


    test_df = pd.DataFrame()
    train_df = pd.DataFrame()

    labels = {"OV": 1, "CV": 1, "rest": 2, "RW": 0, "MEMa1": 2}

    train_files =  []
    for train_subject in train_subjects_list:
        train_files += glob.glob(os.path.join(preprocessed_path, train_subject, "**", "*_test.nc"), recursive=True)
    train_labels = []
    for f in train_files:
        if os.path.basename(f).endswith("_test.nc"):
            train_labels.append(labels[os.path.basename(f).split("_")[-3]])
        else:
            train_labels.append(labels[os.path.basename(f).split("_")[-2]])
    train_df = pd.DataFrame({
        "snirf_file": train_files,
        "trial_type": train_labels})
    
    test_files =  []
    for test_subject in test_subjects_list:
        test_files += glob.glob(os.path.join(preprocessed_path, test_subject, "**", "*_test.nc"), recursive=True)
    test_labels = []
    for f in test_files:
        if os.path.basename(f).endswith("_test.nc"):
            test_labels.append(labels[os.path.basename(f).split("_")[-3]])
        else:
            test_labels.append(labels[os.path.basename(f).split("_")[-2]])
    test_df = pd.DataFrame({
        "snirf_file": test_files,
        "trial_type": test_labels})

    train_df = train_df[train_df['trial_type'] != 2]
    test_df = test_df[test_df['trial_type'] != 2]

    # Save the test and train DataFrames to CSV files
    test_df.to_csv(os.path.join(preprocessed_path, "test_segments.csv"), index=False)
    train_df.to_csv(os.path.join(preprocessed_path, "train_segments.csv"), index=False)
    return os.path.join(preprocessed_path, "train_segments.csv"), os.path.join(preprocessed_path, "test_segments.csv")
