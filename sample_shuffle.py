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
    sequences_extracted = []
    with open(source + '/' + source.split('_')[0] + '_train') as train, open(source + '/' + source.split('_')[0] + '_test_normal') as test_norm:
        for normal_sequences_file in [train, test_norm]:
            for line in normal_sequences_file:
                sequences_extracted.append(line.strip('\n'))
    num_train_logs = math.ceil(train_ratio * len(sequences_extracted))
    print('Randomly selecting ' + str(num_train_logs) + ' sequences from ' + str(len(sequences_extracted)) + ' normal sequences for training')
    train_sequences = set(random.sample(sequences_extracted, num_train_logs))
    print('Shuffle vector files ...')
    with open(source + '/' + source.split('_')[0] + '_train', 'w+') as train, open(source + '/' + source.split('_')[0] + '_test_normal', 'w+') as test_norm:
        for sequence in sequences_extracted:
            if sequence in train_sequences:
                train.write(sequence + '\n')
            else:
                test_norm.write(sequence + '\n')

if __name__ == "__main__":
    do_sample(source, train_ratio)
