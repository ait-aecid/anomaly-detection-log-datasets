from collections import Counter
import argparse
from datetime import datetime

parser = argparse.ArgumentParser()

parser.add_argument("--data_dir", default="hdfs_xu", help="path to input files", type=str, choices=['hdfs_logdeep', 'hdfs_xu', 'hdfs_loghub', 'bgl_loghub', 'bgl_cfdr', 'openstack_loghub', 'hadoop_loghub', 'thunderbird_cfdr', 'adfa_verazuo'])
parser.add_argument("--show_samples", default=10, help="number of samples shown in output", type=int)

params = vars(parser.parse_args())
source = params["data_dir"]
show_samples = params["show_samples"]

sequences_extracted = {}
labels = {}

filename = source.split('_')[0]

with open(source + '/' + filename + '_train') as train, open(source + '/' + filename + '_test_normal') as normal, open(source + '/' + filename + '_test_abnormal') as abnormal:
    print('Load parsed sequences ...')
    line_cnt = 0
    for input_file in [train, normal, abnormal]:
        for line in input_file:
            line_cnt += 1
            if ',' in line:
                parts = line.strip('\n').split(',')
                seq_id = parts[0]
            else:
                parts = [None, line.strip('\n')]
                seq_id = line_cnt
            if seq_id in sequences_extracted:
                print('WARNING: Sequence ID ' + str(seq_id) + ' occurs multiple times in input data!')
            sequences_extracted[seq_id] = parts[1].split(' ')
            if input_file == abnormal:
                labels[seq_id] = 1
            else:
                labels[seq_id] = 0
    num_lines_normal = 0
    num_lines_anomalous = 0
    event_types_normal = set()
    event_types_anomalous = set()
    for seq_id, seq in sequences_extracted.items():
        if labels[seq_id] == 1:
            num_lines_anomalous += len(seq)
            event_types_anomalous.update(seq)
        else:
            num_lines_normal += len(seq)
            event_types_normal.update(seq)
    print('Parsed lines total: ' + str(num_lines_normal + num_lines_anomalous))
    print('Parsed lines normal: ' + str(num_lines_normal) + ' (' + str(round(num_lines_normal * 100.0 / (num_lines_normal + num_lines_anomalous), 1)) + '%)')
    print('Parsed lines anomalous: ' + str(num_lines_anomalous) + ' (' + str(round(num_lines_anomalous * 100.0 / (num_lines_normal + num_lines_anomalous), 1)) + '%)')
    print('Event types total: ' + str(len(set(list(event_types_normal) + list(event_types_anomalous)))))
    print('Event types normal: ' + str(len(event_types_normal)) + ' (' + str(round(len(event_types_normal) * 100.0 / (len(set(list(event_types_normal) + list(event_types_anomalous)))), 1)) + '%)')
    print('Event types anomalous: ' + str(len(event_types_anomalous)) + ' (' + str(round(len(event_types_anomalous) * 100.0 / (len(set(list(event_types_normal) + list(event_types_anomalous)))), 1)) + '%)')
    print('Sequences total: ' + str(len(sequences_extracted)))
    seq_normal = 0
    seq_anomalous = 0
    seq_normal_unique = set()
    seq_anomalous_unique = set()
    seq_normal_cnt = {}
    seq_anomalous_cnt = {}
    count_unique = set()
    count_normal_unique = {}
    count_anomalous_unique = {}
    for seq_id, events in sequences_extracted.items():
        counter_dict_tuple = tuple(sorted(dict(Counter(events)).items()))
        count_unique.add(counter_dict_tuple)
        events_tuple = tuple(events)
        if seq_id not in labels:
            print('WARNING: ' + str(seq_id) + ' not found in labels file!')
        elif labels[seq_id] == 0:
            seq_normal += 1
            seq_normal_unique.add(events_tuple)
            if events_tuple not in seq_normal_cnt:
                seq_normal_cnt[events_tuple] = 1
            else:
                seq_normal_cnt[events_tuple] += 1
            if counter_dict_tuple not in count_normal_unique:
                count_normal_unique[counter_dict_tuple] = 1
            else:
                count_normal_unique[counter_dict_tuple] += 1
        elif labels[seq_id] == 1:
            seq_anomalous += 1
            seq_anomalous_unique.add(events_tuple)
            if events_tuple not in seq_anomalous_cnt:
                seq_anomalous_cnt[events_tuple] = 1
            else:
                seq_anomalous_cnt[events_tuple] += 1
            if counter_dict_tuple not in count_anomalous_unique:
                count_anomalous_unique[counter_dict_tuple] = 1
            else:
                count_anomalous_unique[counter_dict_tuple] += 1
    print('Sequences normal: ' + str(seq_normal) + ' (' + str(round(seq_normal * 100.0 / len(sequences_extracted), 1)) + '%)')
    print('Sequences anomalous: ' + str(seq_anomalous) + ' (' + str(round(seq_anomalous * 100.0 / len(sequences_extracted), 1)) + '%)')
    print('Unique sequences: ' + str(len(set(list(seq_normal_unique) + list(seq_anomalous_unique)))) + ' (' + str(round(len(set(list(seq_normal_unique) + list(seq_anomalous_unique))) * 100.0 / len(sequences_extracted), 1)) + '%)')
    print('Unique sequences normal: ' + str(len(seq_normal_unique)) + ' (' + str(round(len(seq_normal_unique) * 100.0 / len(set(list(seq_normal_unique) + list(seq_anomalous_unique))), 1)) + '%)')
    print('Unique sequences anomalous: ' + str(len(seq_anomalous_unique)) + ' (' + str(round(len(seq_anomalous_unique) * 100.0 / len(set(list(seq_normal_unique) + list(seq_anomalous_unique))), 1)) + '%)')
    normal_in_anomalous = 0
    normal_in_anomalous_unique = 0
    for seq, cnt in seq_normal_cnt.items():
        if seq in seq_anomalous_cnt:
            normal_in_anomalous += cnt
            normal_in_anomalous_unique += 1
    anomalous_in_normal = 0
    anomalous_in_normal_unique = 0
    for seq, cnt in seq_anomalous_cnt.items():
        if seq in seq_normal_cnt:
            anomalous_in_normal += cnt
            anomalous_in_normal_unique += 1
    print('Sequences labeled normal that also occur as anomalous: ' + str(normal_in_anomalous) + ' (' + str(round(100.0 * normal_in_anomalous / seq_normal, 3)) + '%)' + ', ' + str(normal_in_anomalous_unique) + ' unique')
    print('Sequences labeled anomalous that also occur as normal: ' + str(anomalous_in_normal) + ' (' + str(round(100.0 * anomalous_in_normal / seq_anomalous, 3)) + '%)' + ', ' + str(anomalous_in_normal_unique) + ' unique')
    print('Common normal sequences: ')
    cnt_sorted = {k: v for k, v in sorted(seq_normal_cnt.items(), key=lambda item: item[1], reverse=True)}
    i = 0
    for events, info in cnt_sorted.items():
        i += 1
        if i > show_samples:
            break
        print(str(info) + ': ' + str(events))
    print('Common anomalous sequences: ')
    cnt_sorted = {k: v for k, v in sorted(seq_anomalous_cnt.items(), key=lambda item: item[1], reverse=True)}
    i = 0
    for events, info in cnt_sorted.items():
        i += 1
        if i > show_samples:
            break
        print(str(info) + ': ' + str(events))
    print('Unique count vectors: ' + str(len(count_unique)) + ' (' + str(round(len(count_unique) * 100.0 / len(set(list(seq_normal_unique) + list(seq_anomalous_unique))), 1)) + '% of unique sequences or ' + str(round(len(count_unique) * 100.0 / len(sequences_extracted), 1)) + '% of all sequences)')
    print('Unique count vectors normal: ' + str(len(count_normal_unique)) + ' (' + str(round(len(count_normal_unique) * 100.0 / len(count_unique), 1)) + '%)')
    print('Unique count vectors anomalous: ' + str(len(count_anomalous_unique)) + ' (' + str(round(len(count_anomalous_unique) * 100.0 / len(count_unique), 1)) + '%)')
    normal_in_anomalous = 0
    normal_in_anomalous_unique = 0
    for vec, cnt in count_normal_unique.items():
        if vec in count_anomalous_unique:
            normal_in_anomalous += cnt
            normal_in_anomalous_unique += 1
    anomalous_in_normal = 0
    anomalous_in_normal_unique = 0
    for vec, cnt in count_anomalous_unique.items():
        if vec in count_normal_unique:
            anomalous_in_normal += cnt
            anomalous_in_normal_unique += 1
    print('Count vectors labeled normal that also occur as anomalous: ' + str(normal_in_anomalous) + ' (' + str(round(100.0 * normal_in_anomalous / seq_normal, 3)) + '%)' + ', ' + str(normal_in_anomalous_unique) + ' unique')
    print('Count vectors labeled anomalous that also occur as normal: ' + str(anomalous_in_normal) + ' (' + str(round(100.0 * anomalous_in_normal / seq_anomalous, 3)) + '%)' + ', ' + str(anomalous_in_normal_unique) + ' unique')
    print('Common normal count vectors:')
    cnt_sorted = {k: v for k, v in sorted(count_normal_unique.items(), key=lambda item: item[1], reverse=True)}
    i = 0
    for events, info in cnt_sorted.items():
        i += 1
        if i > show_samples:
            break
        print(str(info) + ': ' + str(events))
    print('Common anomalous count vectors:')
    cnt_sorted = {k: v for k, v in sorted(count_anomalous_unique.items(), key=lambda item: item[1], reverse=True)}
    i = 0
    for events, info in cnt_sorted.items():
        i += 1
        if i > show_samples:
            break
        print(str(info) + ': ' + str(events))
