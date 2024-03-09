from collections import Counter
import random
import argparse
import math

parser = argparse.ArgumentParser()

parser.add_argument("--data_dir", default="hdfs_xu", help="path to input files", type=str, choices=['adfa_verazuo', 'hdfs_xu', 'hdfs_loghub', 'bgl_loghub', 'bgl_cfdr', 'openstack_loghub', 'openstack_parisakalaki', 'hadoop_loghub', 'thunderbird_cfdr', 'awsctd_djpasco'])
parser.add_argument("--train_ratio", default=0.01, help="fraction of normal data used for training", type=float)
parser.add_argument("--time_window", default=None, help="size of the fixed time window in seconds (setting this parameter replaces session-based with window-based grouping)", type=float)
parser.add_argument("--sample_ratio", default=1.0, help="fraction of data sampled from normal and anomalous events", type=float)
parser.add_argument("--sorting", default="random", help="sorting mode: pick sequences randomly (random) or only pick the first ones (chronological)", type=str, choices=['random', 'chronological'])
parser.add_argument("--anomaly_types", default="False", help="set to True to additionally create sequence files for each anomaly type (files are named <dataset>_test_abnormal_<anomaly>", type=str, choices=['True', 'False'])

params = vars(parser.parse_args())
source = params["data_dir"]
train_ratio = params["train_ratio"]
tw = params["time_window"]
sample_ratio = params["sample_ratio"]
sorting = params["sorting"]
output_anomaly_types = params["anomaly_types"]

if source in ['adfa_verazuo', 'hdfs_xu', 'hdfs_loghub', 'openstack_loghub', 'openstack_parisakalaki', 'hadoop_loghub', 'awsctd_djpasco'] and tw is not None:
    # Only BGL and Thunderbird should be used with time-window based grouping
    print('WARNING: Using time-window grouping, even though session-based grouping is recommended for this data set.')

def do_sample(source, train_ratio, sorting="random", tw=None):
    header = True
    sequences_extracted = {}
    tw_groups = {} # Only used for time-window based grouping
    tw_labels = {} # Only used for time-window based grouping
    labels = {}
    with open(source + '/parsed.csv') as extracted, open(source + '/' + source.split('_')[0] + '_train', 'w+') as train, open(source + '/' + source.split('_')[0] + '_test_normal', 'w+') as test_norm, open(source + '/' + source.split('_')[0] + '_test_abnormal', 'w+') as test_abnormal:
        cnt = 0
        print('Read in parsed sequences ...')
        for line in extracted:
            if header:
                header = False
                colnames = line.strip('\n').split(';')
                continue
            parts = line.strip('\n').split(';')
            event_id = parts[colnames.index('event_type')]
            if tw is not None:
                # Print processing status
                cnt += 1
                if cnt % 1000000 == 0:
                    print(str(cnt) + ' lines processed, ' + str(len(tw_groups)) + ' time windows found so far')
                # Use label of the event
                label = parts[colnames.index('eventlabel')]
                # Group events by occurrence time
                time = float(parts[colnames.index('time')])
                time_group = math.floor(time / tw) * tw
                if time_group not in tw_groups:
                    tw_groups[time_group] = [event_id]
                else:
                    tw_groups[time_group].append(event_id)
                if time_group not in tw_labels:
                    tw_labels[time_group] = label
                if label != "Normal":
                    # If any event in the time window is anomalous, consider the entire time window as anomalous
                    tw_labels[time_group] = label
            else:
                # Print processing status
                cnt += 1
                if cnt % 1000000 == 0:
                    num_seq_anom = 0
                    for lbl, seqs in sequences_extracted.items():
                        if lbl == 'Normal':
                            continue
                        num_seq_anom += len(seqs)
                    num_normal = 0
                    if 'Normal' in sequences_extracted:
                        num_normal = len(sequences_extracted['Normal'])
                    print(str(cnt) + ' lines processed, ' + str(num_normal) + ' normal and ' + str(num_seq_anom) + ' anomalous sequences found so far')
                # Use label of the entire sequence
                label = parts[colnames.index('label')]
                # Group events by sequence identifier
                seq_id = parts[colnames.index('seq_id')]
                if label not in sequences_extracted:
                    sequences_extracted[label] = {}
                if seq_id not in sequences_extracted[label]:
                    sequences_extracted[label][seq_id] = [event_id]
                else:
                    sequences_extracted[label][seq_id].append(event_id)
        if tw is not None:
            # After processing all lines it is known which label applies to which time window
            for time_group, event_sequence in tw_groups.items():
                if tw_labels[time_group] not in sequences_extracted:
                    sequences_extracted[tw_labels[time_group]] = {}
                sequences_extracted[tw_labels[time_group]][time_group] = event_sequence
        num_seq_anom = 0
        for lbl, seqs in sequences_extracted.items():
            if lbl == 'Normal':
                continue
            num_seq_anom += len(seqs)
        num_normal = 0
        if 'Normal' in sequences_extracted:
            num_normal = len(sequences_extracted['Normal'])
        print('Processing complete, found ' + str(num_normal) + ' normal and ' + str(num_seq_anom) + ' anomalous sequences')
        if sample_ratio < 1:
            sampled_sequences = {}
            num_sampled_anom = 0
            for lbl, seqs in sequences_extracted.items():
                sampled_seq_list = random.sample(list(sequences_extracted[lbl].keys()), math.ceil(sample_ratio * len(seqs)))
                sampled_sequences[lbl] = {}
                for selected_seq in sampled_seq_list:
                    sampled_sequences[lbl][selected_seq] = sequences_extracted[lbl][selected_seq]
                if lbl != "Normal":
                    num_sampled_anom += len(sampled_seq_list)
            sequences_extracted = sampled_sequences
            print('Sampled ' + str(len(sequences_extracted['Normal'])) + ' normal and ' + str(num_sampled_anom) + ' anomalous sequences')
        num_train_logs = math.ceil(train_ratio * len(sequences_extracted['Normal']))
        if sorting == "random":
            print('Randomly selecting ' + str(num_train_logs) + ' sequences from ' + str(len(sequences_extracted['Normal'])) + ' normal sequences for training')
            train_seq_id_list = set(random.sample(list(sequences_extracted['Normal'].keys()), num_train_logs))
        elif sorting == "chronological":
            print('Chronologically selecting ' + str(num_train_logs) + ' sequences from ' + str(len(sequences_extracted['Normal'])) + ' normal sequences for training')
            train_seq_id_list = set(list(sequences_extracted['Normal'].keys())[:num_train_logs])
        else:
            print("Warning: Unknown sorting mode!")
        print('Write vector files ...')
        for label, seq_id_dict in sequences_extracted.items():
            if label == 'Normal':
                for seq_id, event_list in seq_id_dict.items():
                    if seq_id in train_seq_id_list:
                        train.write(str(seq_id) + ',' + ' '.join([str(event) for event in event_list]) + '\n')
                    else:
                        test_norm.write(str(seq_id) + ',' + ' '.join([str(event) for event in event_list]) + '\n')
            elif label == "Anomaly" or output_anomaly_types == "False":
                for seq_id, event_list in seq_id_dict.items():
                    test_abnormal.write(str(seq_id) + ',' + ' '.join([str(event) for event in event_list]) + '\n')
            else:
                with open(source + '/' + source.split('_')[0] + '_test_abnormal_' + label, 'w+') as test_label:
                    for seq_id, event_list in seq_id_dict.items():
                        test_label.write(str(seq_id) + ',' + ' '.join([str(event) for event in event_list]) + '\n')
                        test_abnormal.write(str(seq_id) + ',' + ' '.join([str(event) for event in event_list]) + '\n')

if __name__ == "__main__":
    do_sample(source, train_ratio, sorting, tw)
