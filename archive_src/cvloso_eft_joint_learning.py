import pickle
import numpy as np
from sklearn.metrics import f1_score
import torch
import os
from torch import nn, optim
from datetime import datetime
from pathlib import Path

from models import CNN2D_BaselineV2, CNN2DModel, Dataset, PreloadedDataset


DATASET_ROOT = Path("datasets/processed")
DENSE_DATASET_NAME = "BallSqueezingHD_modified"
AUGMENT_DATASET_NAME = "FreshMotor"
AUGMENT_FREQS = {0.5, 0.7, 1.0}
AUGMENT_RATIO = 1.0
AUGMENT_SEED = 1337
METRIC_SPLITS = ('train', 'val', 'test')


def build_model(model_spec):
    return model_spec['builder']()


def make_dataset(file_paths, preload):
    """Create dataset, optionally preloading all data into RAM."""
    if isinstance(file_paths, np.ndarray):
        file_paths = file_paths.tolist()
    file_paths = list(file_paths)

    if preload:
        return PreloadedDataset(file_paths, pin_memory=True)
    return Dataset(file_paths)


def build_loader(file_paths, batch_size, shuffle=True, preload=True):
    """Create DataLoader for the given file paths."""
    dataset = make_dataset(file_paths, preload=preload)
    params = {
        'batch_size': batch_size,
        'shuffle': shuffle,
        'num_workers': 0 if preload else 4,
        'pin_memory': True,
    }
    return torch.utils.data.DataLoader(dataset, **params)


def collect_paths(meta_events_by_freq, freq_values, excluded_subject):
    """Collect file paths for the requested frequencies, skipping the held-out subject."""
    collected = []
    for frequency in freq_values:
        for subject, entries in meta_events_by_freq.get(frequency, {}).items():
            if subject == excluded_subject:
                continue
            collected.extend(entries)
    return collected


def load_meta_events(dataset_name, freqs):
    """Load metadata dictionaries for the requested frequencies."""
    meta_events = {}
    base_dir = DATASET_ROOT / dataset_name
    for freq in freqs:
        freq_str = str(freq)
        meta_path = base_dir / f"frq{freq_str}" / f"meta_event_{freq_str}.pkl"
        with open(meta_path, 'rb') as handle:
            meta_events[freq] = pickle.load(handle)
    return meta_events


def load_files_to_sessions(dataset_name):
    """Load the files-to-sessions mapping for the given dataset."""
    mapping_path = DATASET_ROOT / dataset_name / "files_to_sessions.pkl"
    with open(mapping_path, 'rb') as handle:
        return pickle.load(handle)


def collect_subject_files(meta_events, files_to_sessions, allowed_sessions=None):
    """Gather file paths per subject, optionally filtering by allowed sessions."""
    subject_files = {}
    for freq_dict in meta_events.values():
        for subject, files in freq_dict.items():
            for file_path in files:
                run_name = files_to_sessions.get(file_path)
                if run_name is None:
                    continue
                if allowed_sessions and run_name not in allowed_sessions:
                    continue
                subject_files.setdefault(subject, []).append(file_path)
    return subject_files


def stratified_subject_sample(subject_files, ratio, rng):
    """Sample approximately ratio fraction of files per subject."""
    if ratio <= 0 or not subject_files:
        return [], {}

    total_candidates = sum(len(samples) for samples in subject_files.values())
    if total_candidates == 0:
        return [], {}
    if ratio >= 1:
        return [fp for samples in subject_files.values() for fp in samples], {sub: len(files) for sub, files in subject_files.items()}

    pools = {subject: rng.permutation(files).tolist() for subject, files in subject_files.items() if files}
    if not pools:
        return [], {}

    target = max(1, min(total_candidates, int(round(total_candidates * ratio))))
    subject_ids = list(pools.keys())
    rng.shuffle(subject_ids)
    base_quota = target // len(subject_ids)
    selected = []
    counts = {subject: 0 for subject in subject_ids}

    for subject in subject_ids:
        pool = pools[subject]
        take = min(base_quota, len(pool))
        if take:
            selected.extend(pool[:take])
            counts[subject] += take
            pools[subject] = pool[take:]

    remaining = target - sum(counts.values())
    while remaining > 0:
        available = [subject for subject in subject_ids if pools[subject]]
        if not available:
            break
        rng.shuffle(available)
        for subject in available:
            if remaining == 0:
                break
            pool = pools[subject]
            if not pool:
                continue
            selected.append(pool.pop())
            counts[subject] += 1
            remaining -= 1
            if remaining == 0:
                break

    counts = {sub: cnt for sub, cnt in counts.items() if cnt > 0}
    return selected, counts


def _extract_file_info(file_path, files_to_sessions):
    """Return subject id, run name, and frequency tag for a sample path."""
    path_str = str(file_path)
    norm_path = os.path.normpath(path_str)
    parts = norm_path.split(os.sep)

    freq = next((p for p in parts if p.startswith('frq')), 'unknown_freq')
    subject = next((p for p in parts if p.startswith('sub-')), 'unknown_subject')

    run = files_to_sessions.get(path_str)
    if run is None:
        run = files_to_sessions.get(path_str.replace(os.sep, '/'), 'unknown_run')

    return subject, run, freq


def summarize_split(file_paths, files_to_sessions):
    """Summarize subject/run/frequency usage for a split."""
    summary = {}
    for file_path in file_paths:
        subject, run, freq = _extract_file_info(file_path, files_to_sessions)
        subj_entry = summary.setdefault(subject, {})
        subj_entry.setdefault(freq, set()).add(run)
    return summary


def format_summary_lines(split_name, summary, sample_count):
    lines = [f"{split_name} ({sample_count} samples):"]
    if not summary:
        lines.append("  - (no samples)")
        return lines

    for subject in sorted(summary):
        freq_chunks = []
        for freq in sorted(summary[subject]):
            runs = sorted(summary[subject][freq])
            freq_chunks.append(f"{freq} [{', '.join(runs)}]")
        lines.append(f"  - {subject}: " + "; ".join(freq_chunks))
    return lines


def append_log(log_path, lines):
    with open(log_path, 'a') as handle:
        handle.write("\n".join(lines))
        handle.write("\n")




def main():
    model_variants = {
        'baseline': {
            'label': 'CNN2D_BaselineV2',
            'builder': CNN2D_BaselineV2,
        },
        'cnn2d': {
            'label': 'CNN2DModel',
            'builder': lambda: CNN2DModel(num_classes=2),
        },
    }

    
    # Base split is built from all freq=0.5 runs; optional sets control extra datasets appended afterward.
    base_split_freq = 0.5
    extra_train_freqs = {0.7, 1.0} # or set()
    extra_val_freqs = set()  # add {0.7} to append frq0.7 samples to the val split
    test_freq = 0.5

    val_split_ratio = 0.33
    split_seed = 42
    
    lr = 1e-5
    batch_size = 32
    epochs = 200
    use_preloaded_dataset = False
    
    dataset_name = DENSE_DATASET_NAME
    augment_dataset_name = AUGMENT_DATASET_NAME if AUGMENT_RATIO > 0 else None
    dataset_components = [dataset_name]
    if augment_dataset_name:
        dataset_components.append(augment_dataset_name)
    dataset_names_tag = "__".join(dataset_components)
    aug_pct = int(round(max(0.0, min(AUGMENT_RATIO, 1.0)) * 100))
    augment_ratio_tag = f"aug{aug_pct:02d}%"
    selected_model_key = 'baseline'

    model_spec = model_variants[selected_model_key]
    base, exp = f"{lr:.0e}".split("e")
    lr_tag = f"lr{base}e{int(exp)}"
    model_name = f"{model_spec['label']}_{lr_tag}"
    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{model_name}__{augment_ratio_tag}__{run_tag}"

    training_root = os.path.join("training", dataset_names_tag, run_name)
    models_dir = os.path.join(training_root, 'models')
    loss_dir = os.path.join(training_root, 'loss')
    log_dir = os.path.join(training_root, 'logs')
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(loss_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    split_log_path = os.path.join(log_dir, f"split_usage_{run_name}.log")

    append_log(
        split_log_path,
        [
            "",
            f"=== Run started {datetime.now().isoformat()} ===",
            f"Model: {model_name}",
            f"Run tag: {run_tag}",
            f"Datasets: {dataset_names_tag}",
            f"Base split freq: frq{base_split_freq}",
            f"Extra train freq tags: {sorted(f'frq{f}' for f in extra_train_freqs) if extra_train_freqs else 'none'}",
            f"Extra val freq tags: {sorted(f'frq{f}' for f in extra_val_freqs) if extra_val_freqs else 'none'}",
            f"Test freq: frq{test_freq}",
            f"Validation split ratio: {val_split_ratio}",
            f"Validation split seed: {split_seed}",
            f"Augmentation dataset: {augment_dataset_name or 'none'}",
            f"Augmentation ratio: {AUGMENT_RATIO if augment_dataset_name else 0}",
            f"Augmentation frequencies: {sorted(f'frq{f}' for f in AUGMENT_FREQS) if AUGMENT_FREQS else 'all'}",
        ],
    )

    files_to_sessions = load_files_to_sessions(dataset_name)
    all_files_to_sessions = dict(files_to_sessions)

    freqs_to_load = sorted({base_split_freq, test_freq} | extra_train_freqs | extra_val_freqs)
    meta_events_by_freq = load_meta_events(dataset_name, freqs_to_load)

    augment_subset = []
    if augment_dataset_name:
        augment_freqs_to_load = sorted(AUGMENT_FREQS) if AUGMENT_FREQS else []
        if not augment_freqs_to_load:
            raise ValueError("Augmentation requires at least one frequency.")
        augment_meta_events = load_meta_events(augment_dataset_name, augment_freqs_to_load)
        augment_files_to_sessions = load_files_to_sessions(augment_dataset_name)
        all_files_to_sessions.update(augment_files_to_sessions)
        augment_subject_files = collect_subject_files(augment_meta_events, augment_files_to_sessions)
        augment_rng = np.random.default_rng(AUGMENT_SEED)
        augment_subset, _ = stratified_subject_sample(
            augment_subject_files,
            AUGMENT_RATIO,
            augment_rng,
        )

    # train-data
    subjects = list(meta_events_by_freq.get(base_split_freq, {}).keys())
    if not subjects:
        raise ValueError(f"No subjects available for base split frequency frq{base_split_freq}.")

    for REMOVE_SUB in subjects:  # iterate over the subjects in a LOSO evaluation.
        base_candidates = collect_paths(meta_events_by_freq, {base_split_freq}, REMOVE_SUB)
        if not base_candidates:
            raise ValueError(
                f"No frq{base_split_freq} samples available for LOSO hold-out subject '{REMOVE_SUB}'."
            )

        # Shuffle the base-frequency pool once per subject so split is reproducible with the seed.
        rng = np.random.default_rng(split_seed)
        perm = rng.permutation(len(base_candidates))

        if len(base_candidates) > 1:
            val_count = int(len(base_candidates) * val_split_ratio)
            val_count = max(1, min(val_count, len(base_candidates) - 1))
        else:
            val_count = len(base_candidates)

        # Keep 70/30 split of the shuffled base-frequency samples as training/validation anchors.
        base_val = [base_candidates[i] for i in perm[:val_count]]
        base_train = [base_candidates[i] for i in perm[val_count:]]

        train_extra = collect_paths(meta_events_by_freq, extra_train_freqs, REMOVE_SUB)
        val_extra = collect_paths(meta_events_by_freq, extra_val_freqs, REMOVE_SUB)

        train_data = np.array(base_train + train_extra + augment_subset, dtype=object)
        validation_data = np.array(base_val + val_extra, dtype=object)

        test_meta = meta_events_by_freq[test_freq]
        test_data = test_meta[REMOVE_SUB]

        train_summary = summarize_split(train_data, all_files_to_sessions)
        val_summary = summarize_split(validation_data, all_files_to_sessions)
        test_summary = summarize_split(test_data, all_files_to_sessions)

        split_lines = [
            "",
            f"--- LOSO hold-out subject: {REMOVE_SUB} ---",
            f"Base freq frq{base_split_freq}: {len(base_candidates)} total -> "
            f"{len(base_train)} train / {len(base_val)} val",
            f"Extra appended: {len(train_extra)} train from "
            f"{sorted(f'frq{f}' for f in extra_train_freqs) or 'none'}; "
            f"{len(val_extra)} val from {sorted(f'frq{f}' for f in extra_val_freqs) or 'none'}",
            f"Augmentation dataset: {augment_dataset_name or 'none'} -> {len(augment_subset)} samples (ratio {AUGMENT_RATIO})",
            *format_summary_lines("Train split", train_summary, len(train_data)),
            *format_summary_lines("Validation split", val_summary, len(validation_data)),
            *format_summary_lines("Test split", test_summary, len(test_data)),
        ]

        print("\n".join(split_lines))
        append_log(split_log_path, split_lines)

        print('train on:', train_data.shape[0], 'validate on:', validation_data.shape[0],  'test on:', len(test_data))
                
        print('CUDA:', torch.cuda.is_available())

        # Dataloaders (preloaded by default for speed)
        training_generator = build_loader(train_data, batch_size=batch_size, shuffle=True, preload=use_preloaded_dataset)
        validation_generator = build_loader(validation_data, batch_size=batch_size, shuffle=True, preload=use_preloaded_dataset)
        testing_generator = build_loader(test_data, batch_size=batch_size, shuffle=True, preload=use_preloaded_dataset)

        # model
        model = build_model(model_spec)
        print('number of parameters:', sum(p.numel() for p in model.parameters() if p.requires_grad))

        model.cuda()
        optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5, betas=(0.9, 0.999))
        loss_function = nn.CrossEntropyLoss()

        mats = []
        f1_max = -np.inf
        for epoch in range(epochs):
            print('\nEpoch-----------{}------{}'.format(epoch, model_name))

            model.train()
            train_f1, train_loss = [], []
            for it, (x, y) in enumerate(training_generator):
                optim.zero_grad()
                
                #x = x.flatten(1, 2)
                y_hat = model(x.cuda())
                loss = loss_function(y_hat, y.cuda())
                loss.backward()
                optim.step()

                f1 = f1_score(y.cpu().numpy(), y_hat.argmax(dim=-1).detach().cpu().numpy(), average='micro')
                train_f1.append(f1)
                train_loss.append(loss.item())
                print('\r[{:04d}] loss: {:.2f} f1-score: {:.2f}'.format(it, loss.item(), f1), end='')

            print('\n-----Train- loss {:.4f} and f1: {:.2f}'.format(np.mean(train_loss), np.mean(train_f1)))

            model.eval()
            val_loss, val_f1= [], []
            for it, (x, y) in enumerate(validation_generator):
                
                #x = x.flatten(1, 2)
                y_hat = model(x.cuda())
                loss = loss_function(y_hat, y.cuda())
                f1 = f1_score(y.cpu().numpy(), y_hat.argmax(dim=-1).detach().cpu().numpy(), average='micro')
                print('\r[{:04d}] validation loss: {:.2f} f1-score: {:.2f}'.format(it,loss.item(), f1), end='')
                val_f1.append(f1)
                val_loss.append(loss.item())
            print('\n-----Validation- loss {:.4f} and f1: {:.2f}'.format(np.mean(val_loss), np.mean(val_f1)))
            
            #save-the-model based on validation data
            if np.mean(val_f1) > f1_max:
                print('the performance increased from:', f1_max, ' to ', np.mean(val_f1))
                f1_max = np.mean(val_f1)
                torch.save(
                    model.state_dict(),
                    os.path.join(models_dir, f"model_{run_name}_{REMOVE_SUB}.tm")
                )

            test_loss, test_f1 = [], []
            for it, (x, y) in enumerate(testing_generator):
                #x = x.flatten(1, 2)
                y_hat = model(x.cuda())
                loss = loss_function(y_hat, y.cuda())
                f1 = f1_score(y.cpu().numpy(), y_hat.argmax(dim=-1).detach().cpu().numpy(), average='micro')
                print('\r[{:04d}] testing loss: {:.2f} f1-score: {:.2f}'.format(it,loss.item(), f1), end='')
                test_f1.append(f1)
                test_loss.append(loss.item())
            print('\n-----LOSO Test- loss {:.4f} and f1: {:.2f}'.format(np.mean(test_loss), np.mean(test_f1)))

            mats.append([np.mean(train_loss), np.mean(train_f1), np.mean(val_loss), np.mean(val_f1), np.mean(test_loss), np.mean(test_f1)])
            np.save(
                os.path.join(loss_dir, f"mats_{run_name}_{REMOVE_SUB}"),
                np.array(mats)
            )

    print('Training completed.')

if __name__ == "__main__":
    main()
