from collections import Counter
import random
import argparse
import math

parser = argparse.ArgumentParser()

parser.add_argument("--data_dir", default="hdfs_xu", help="path to input files", type=str, choices=['adfa_verazuo', 'hdfs_xu', 'hdfs_loghub', 'bgl_loghub', 'bgl_cfdr', 'openstack_loghub', 'openstack_parisakalaki', 'hadoop_loghub', 'thunderbird_cfdr', 'awsctd_djpasco'])
parser.add_argument("--train_ratio", default=0.01, help="fraction of normal data used for training", type=float)

params = vars(parser.parse_args())
source = params["data_dir"]
train_ratio = params["train_ratio"]

def do_sample(source, train_ratio):
    header = True
    sequences_extracted = {}
    labels = {}
    with open(source + '/parsed.csv') as extracted, open(source + '/' + source.split('_')[0] + '_train', 'w+') as train, open(source + '/' + source.split('_')[0] + '_test_normal', 'w+') as test_norm, open(source + '/' + source.split('_')[0] + '_test_abnormal', 'w+') as test_abnormal:
        cnt = 0
        print('Read in parsed sequences ...')
        for line in extracted:
            if header:
                header = False
                colnames = line.strip('\n').split(';')
                continue
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
            parts = line.strip('\n').split(';')
            seq_id = parts[colnames.index('seq_id')]
            event_id = parts[colnames.index('event_type')]
            label = parts[colnames.index('label')]
            if label not in sequences_extracted:
                sequences_extracted[label] = {}
            if seq_id not in sequences_extracted[label]:
                sequences_extracted[label][seq_id] = [event_id]
            else:
                sequences_extracted[label][seq_id].append(event_id)
        num_train_logs = math.ceil(train_ratio * len(sequences_extracted['Normal']))
        print('Randomly selecting ' + str(num_train_logs) + ' sequences from ' + str(len(sequences_extracted['Normal'])) + ' normal sequences for training')
        train_seq_id_list = random.sample(list(sequences_extracted['Normal'].keys()), num_train_logs)
        print('Write vector files ...')
        cnt = 0
        for label, seq_id_dict in sequences_extracted.items():
            cnt += 1
            if cnt % 10 == 0:
                print(str(cnt) + ' sequences written')
            if label == 'Normal':
                for seq_id, event_list in seq_id_dict.items():
                    if seq_id in train_seq_id_list:
                        train.write(str(seq_id) + ',' + ' '.join([str(event) for event in event_list]) + '\n')
                    else:
                        test_norm.write(str(seq_id) + ',' + ' '.join([str(event) for event in event_list]) + '\n')
            elif label == "Anomaly":
                for seq_id, event_list in seq_id_dict.items():
                    test_abnormal.write(str(seq_id) + ',' + ' '.join([str(event) for event in event_list]) + '\n')
            else:
                with open(source + '/' + source.split('_')[0] + 'test_abnormal_' + label, 'w+') as test_label:
                    for seq_id, event_list in seq_id_dict.items():
                        test_label.write(str(seq_id) + ',' + ' '.join([str(event) for event in event_list]) + '\n')
                        test_abnormal.write(str(seq_id) + ',' + ' '.join([str(event) for event in event_list]) + '\n')

if __name__ == "__main__":
    do_sample(source, train_ratio)
