import pickle
import numpy as np
from sklearn.metrics import f1_score
import torch
import os
from torch import nn, optim
from datetime import datetime

from models import CNN2D_BaselineV2, CNN2DModel, Dataset, PreloadedDataset


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

    # train_runs = {'run-1', 'run-2'}
    # val_runs = {'run-3'}

    # train_freqs = {0.2, 0.5, 0.7}
    # val_freqs = {0.5}
    # test_freq = 0.5
    
    # train_freqs = {0.2, 0.5, 0.7}
    # val_freqs = {0.2, 0.5, 0.7}
    # test_freq = 0.5
    
    # train_freqs = {0.5}
    # val_freqs = {0.5}
    # test_freq = 0.5

    # train_runs = {'run-1', 'run-2', 'run-3'}
    # val_runs = {'run-3'}
    # train_freqs = {0.5}
    # val_freqs = {0.5}
    # test_freq = 0.5
    
    train_runs = {'run-2', 'run-3'}
    val_runs = {'run-1'}
    train_freqs = {0.2, 0.5, 0.7}
    val_freqs = {0.5}
    test_freq = 0.5

    
    lr = 1e-4
    batch_size = 32
    use_preloaded_dataset = False
    
    dataset_name = "BallSqueezingHD_modified"
    # dataset_name = "FreshMotor"
    selected_model_key = 'baseline'

    model_spec = model_variants[selected_model_key]
    base, exp = f"{lr:.0e}".split("e")
    lr_tag = f"lr{base}e{int(exp)}"
    model_name = f"{model_spec['label']}_{lr_tag}"
    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{model_name}__{run_tag}"

    training_root = os.path.join("training", dataset_name, run_name)
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
            f"Dataset: {dataset_name}",
            f"Train runs: {sorted(train_runs)}",
            f"Validation runs: {sorted(val_runs)}",
            f"Train freqs: {sorted(train_freqs)}",
            f"Validation freqs: {sorted(val_freqs)}",
            f"Test freq: {test_freq}",
        ],
    )

    events = os.path.join(
        "datasets/processed",
        dataset_name,
        "frq{}",
        "meta_event_{}.pkl",
    )

    files_to_sessions_name = os.path.join(
        "datasets/processed",
        dataset_name,
        "files_to_sessions.pkl",
    )

    freqs_to_load = sorted(set(train_freqs) | set(val_freqs) | {test_freq})
    train_freq_tags = {f"frq{f}" for f in train_freqs}
    val_freq_tags = {f"frq{f}" for f in val_freqs}

    meta_events_by_freq = {}
    for freq in freqs_to_load:
        with open(events.format(freq, freq), 'rb') as handle:
            meta_events_by_freq[freq] = pickle.load(handle)

    # train-data
    subjects = list(meta_events_by_freq[freqs_to_load[0]].keys())
    for REMOVE_SUB in subjects: #iterate over the subjects in a LOSO evaluation. 
        candidate_segments = []
        for freq in freqs_to_load:
            meta_dict = meta_events_by_freq[freq]
            for sub in meta_dict:
                if sub != REMOVE_SUB and sub in meta_dict:
                    candidate_segments += meta_dict[sub]
        
        # generate the validation dataset        
        files_to_sessions=None
        with open(files_to_sessions_name, 'rb') as handle:
            files_to_sessions = pickle.load(handle) 

        # Use the configured run sets directly.
        run_set = set(train_runs) | set(val_runs)
        if not run_set:
            raise ValueError("No runs specified in train_runs/val_runs.")

        train_sessions = set(train_runs)
        validation_sessions = set(val_runs)

        train_data_, validation_data = [], []
        for file in candidate_segments:
            run_name = files_to_sessions[file]
            if run_name not in run_set:
                raise ValueError(f"Unknown run '{run_name}' for dataset '{dataset_name}'")
            freq_tag = _extract_file_info(file, files_to_sessions)[2]

            in_train = (run_name in train_sessions) and (freq_tag in train_freq_tags)
            in_val = (run_name in validation_sessions) and (freq_tag in val_freq_tags)

            if in_train:
                train_data_.append(file)
            if in_val:
                validation_data.append(file)
        train_data = np.array(train_data_)
        validation_data = np.array(validation_data)

        test_meta = meta_events_by_freq[test_freq]
        test_data = test_meta[REMOVE_SUB]

        train_summary = summarize_split(train_data, files_to_sessions)
        val_summary = summarize_split(validation_data, files_to_sessions)
        test_summary = summarize_split(test_data, files_to_sessions)

        split_lines = [
            "",
            f"--- LOSO hold-out subject: {REMOVE_SUB} ---",
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
        for epoch in range(50):
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
