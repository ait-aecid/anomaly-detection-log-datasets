import math
import argparse
import Levenshtein
import time

parser = argparse.ArgumentParser()

parser.add_argument("--data_dir", default="hdfs_xu", help="path to input files", type=str, choices=['hdfs_xu', 'hdfs_logdeep', 'hdfs_loghub', 'bgl_loghub', 'bgl_cfdr', 'openstack_loghub', 'openstack_parisakalaki', 'hadoop_loghub', 'thunderbird_cfdr', 'adfa_verazuo', 'awsctd_djpasco'])
parser.add_argument("--time_detection", default="False", help="carry out detection based on interarrival times (requires parsed.csv file)", type=str)
parser.add_argument("--time_window", default=None, help="size of the fixed time window in seconds (only required for time_detection; provide the same time_window that was used when running sample.py)", type=float)

params = vars(parser.parse_args())
data_dir = params["data_dir"]
time_det = params["time_detection"] == "True"
tw = params["time_window"]

def train_cluster_count_vectors(sequences, idf):
    # Learn all unique count vectors from training file
    train_vectors = []
    known_event_types = {}
    idf_weights = {}
    cnt = 0
    for seq_id, sequence in sequences.items():
        cnt += 1
        # Split sequences into single event types
        train_vector = {}
        for part in sequence:
            # Learn all known event types that occur in the training data set
            if part not in known_event_types:
                known_event_types[part] = 1
            else:
                known_event_types[part] += 1
            # Create an event count vector for each sequence
            if part in train_vector:
                train_vector[part] += 1
            else:
                train_vector[part] = 1
            if idf:
                # Count the sequences where each event type occurs in at least once
                if part in idf_weights:
                    idf_weights[part].add(seq_id)
                else:
                    idf_weights[part] = set([seq_id])
        if train_vector not in train_vectors:
            train_vectors.append(train_vector)
    # N stores the total number of sequences
    N = cnt
    for event_type in idf_weights:
        idf_weights[event_type] = math.log10((1 + N) / len(idf_weights[event_type]))
    return train_vectors, known_event_types, idf_weights

def iterate_threshold(dists):
    # Iterate over thresholds between 0 and 1 with step width of 0.01 and create a dictionary of detected samples
    detected_dict = {}
    for i in range(0, 100):
        detected = set()
        threshold = i / 100
        for seq_id, dist in dists.items():
            if dist >= threshold:
                detected.add(seq_id)
        detected_dict[threshold] = detected
    return detected_dict

def test_cluster_count_vectors(train_vectors, test_vectors, normal_seq_ids, abnormal_seq_ids, normalize, idf, idf_weights):
    # Compute minimum count vector distance for every test vector
    dists = {}
    # No need to compute similarity of the same test vectors multiple times
    known_dists = {}
    # No need to compare with the same train vectors multiple times
    train_vectors_set = set()
    for train_vector in train_vectors:
        train_vector_tuple = tuple(sorted(list(train_vector.items())))
        train_vectors_set.add(train_vector_tuple)
    train_vectors_reduced = []
    for train_vector_tuple in train_vectors_set:
        train_vectors_reduced.append(dict(train_vector_tuple))
    for seq_id, test_vector in test_vectors.items():
        test_vector_tuple = tuple(sorted(list(test_vector.items())))
        if test_vector_tuple in known_dists:
            dist = known_dists[test_vector_tuple]
        else:
            dist = check_count_vector(train_vectors_reduced, test_vector, normalize, idf, idf_weights)
            known_dists[test_vector_tuple] = dist
        dists[seq_id] = dist
    # Return detected samples for various thresholds
    return iterate_threshold(dists)

def get_best(detected_dict, detected_additional, normal_seq_ids, abnormal_seq_ids):
    # Find the threshold that maximizes F1 score from a dictionary of detected samples
    best_fone = None
    best_threshold = None
    for threshold, detected in detected_dict.items():
        detected_union = detected.union(detected_additional)
        tp, fn, tn, fp = evaluate(detected_union, normal_seq_ids, abnormal_seq_ids)
        fone = get_fone(tp, fn, tn, fp)
        if best_fone is None or fone > best_fone:
            best_fone = fone
            best_threshold = threshold
    return best_threshold

def test_edit_distance(train_sequences, test_sequences):
    # Compute minimum edit distance for every test sequence
    dists = {}
    dists_by_seq = {} # No need to process the same abnormal sequences multiple times; just remember the dist already computed
    train_sequence_set = set() # No need to compare with the same normal sequence multiple times; remove redundant ones
    train_sequence_unique = []
    for seq_id, train_sequence in train_sequences.items():
        train_sequence_tuple = tuple(train_sequence)
        train_sequence_set.add(train_sequence_tuple)
    train_sequence_unique = list(train_sequence_set) # Transform set to list so that sequences can be reordered
    for seq_id, test_sequence in test_sequences.items():
        test_sequence_tuple = tuple(test_sequence)
        if test_sequence_tuple in dists_by_seq:
            dists[seq_id] = dists_by_seq[test_sequence_tuple]
        else:
            min_dist = 2 # initial value for finding minimum distance (distance is normalized in range 0 to 1)
            for normal_event_sequence in train_sequence_unique:
                norm_fact = max(len(normal_event_sequence), len(test_sequence))
                dist = Levenshtein.distance(normal_event_sequence, test_sequence, score_cutoff=math.floor(norm_fact * min_dist)) / norm_fact
                if dist < min_dist:
                    min_dist = dist
                    best_matching_train_seq = normal_event_sequence
            dists[seq_id] = min_dist
            dists_by_seq[test_sequence_tuple] = min_dist
            # Move sequence with highest similarity to the front of list as it is likely a good match for the following sequences as well
            train_sequence_unique.remove(best_matching_train_seq)
            train_sequence_unique = [best_matching_train_seq] + train_sequence_unique
    # Return detected samples for various thresholds
    return iterate_threshold(dists)

def check_count_vector(train_vectors, test_vector, normalize, idf, idf_weights):
    # Compute distance of test vector to most similar train vector
    min_dist = None
    for train_vector in train_vectors:
        # Iterate over all known count vectors and check if there is at least one that is similar enough to consider the currently processed sequence as normal
        manh = 0
        limit = 0
        for event_type in set(list(train_vector.keys()) + list(test_vector.keys())):
            idf_fact = 1
            if idf:
                if event_type in idf_weights:
                    idf_fact = idf_weights[event_type]
            norm_sum_train = 1
            norm_sum_test = 1
            # Sum up the l1 norm and the highest possible distance for normalization
            if normalize:
                norm_sum_train = sum(train_vector.values())
                norm_sum_test = sum(test_vector.values())
            if event_type not in train_vector:
                manh += test_vector[event_type] * idf_fact / norm_sum_test
                limit += test_vector[event_type] * idf_fact / norm_sum_test
            elif event_type not in test_vector:
                manh += train_vector[event_type] * idf_fact / norm_sum_train
                limit += train_vector[event_type] * idf_fact / norm_sum_train
            else:
                manh += abs(train_vector[event_type] * idf_fact / norm_sum_train - test_vector[event_type] * idf_fact / norm_sum_test)
                limit += max(train_vector[event_type] * idf_fact / norm_sum_train, test_vector[event_type] * idf_fact / norm_sum_test)
        if min_dist is None:
            # Initialize min_dist for first count vector
            min_dist = manh / limit
        else:
            # Update min_dist if a more similar count vector is found
            if manh / limit < min_dist:
                min_dist = manh / limit
    return min_dist

def get_vectors(sequences):
    # Transform sequences into count vectors
    vectors = {}
    for seq_id, sequence in sequences.items():
        vectors[seq_id] = {}
        for part in sequence:
            if part not in vectors[seq_id]:
                vectors[seq_id][part] = 1
            else:
                vectors[seq_id][part] += 1
    return vectors

def get_fone(tp, fn, tn, fp):
    # Compute the F1 score based on detected samples
    if tp + fp + fn == 0:
        return 'inf'
    return tp / (tp + 0.5 * (fp + fn))

def evaluate(detected, normal, abnormal):
    # Count true positives, false negatives, true negatives, and false positives
    tp = len(detected.intersection(abnormal))
    fn = len(set(abnormal).difference(detected))
    tn = len(set(normal).difference(detected))
    fp = len(detected.intersection(normal))
    return tp, fn, tn, fp

def print_results(name, tp, fn, tn, fp, threshold, det_time):
    # Compute metrics and return a dictionary with results
    if tp + fn == 0:
        tpr = "inf"
    else:
        tpr = tp / (tp + fn)
    if fp + tn == 0:
        fpr = "inf"
    else:
        fpr = fp / (fp + tn)
    if tn + fp == 0:
        tnr = "inf"
    else:
        tnr = tn / (tn + fp)
    if tp + fp == 0:
        p = "inf"
    else:
        p = tp / (tp + fp)
    fone = get_fone(tp, fn, tn, fp)
    acc = (tp + tn) / (tp + tn + fp + fn)
    mcc = "inf"
    if tp + fp != 0 and tp + fn != 0 and tn + fp != 0 and tn + fn != 0:
        mcc = ((tp * tn) - (fp * fn)) / math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    print('')
    print(name)
    if threshold is not None:
        print(' Threshold=' + str(threshold))
    else:
        threshold = -1
    print(' Time=' + str(det_time))
    print(' TP=' + str(tp))
    print(' FP=' + str(fp))
    print(' TN=' + str(tn))
    print(' FN=' + str(fn))
    print(' TPR=R=' + str(tpr))
    print(' FPR=' + str(fpr))
    print(' TNR=' + str(tnr))
    print(' P=' + str(p))
    print(' F1=' + str(fone))
    print(' ACC=' + str(acc))
    print(' MCC=' + str(mcc))
    return {'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn, 'tpr': tpr, 'fpr': fpr, 'tnr': tnr, 'p': p, 'f1': fone, 'acc': acc, 'threshold': threshold, 'name': name, 'time': det_time}

read_lines_cnt = 0 # global counter
def read_lines(source):
    # Load sequences from files
    global read_lines_cnt
    sequences = {}
    for line in source:
        read_lines_cnt += 1
        if ',' in line:
            # Parsed sequences where sequence identifier is available (first element of csv)
            parts = line.strip('\n ').split(',')
            sequences[parts[0]] = parts[1].split(' ')
        else:
            # Parsed sequences without sequence identifier (e.g. LogDeep HDFS data set)
            parts = line.strip('\n ').split(' ')
            sequences[read_lines_cnt] = parts
    return sequences

def get_known_event_types(sequences):
    # Return set of event types in training data
    known_event_types = set()
    for seq_id, sequence in sequences.items():
        known_event_types.update(set(sequence))
    return known_event_types

def detect_known_event_types(known_event_types, sequences):
    # Identify all sequences that include at least one event type not known from training sequences
    detected = set()
    for seq_id, sequence in sequences.items():
        for event in sequence:
            if event not in known_event_types:
                detected.add(seq_id)
                break
    return detected

def get_length_range(sequences):
    # Get minimum and maximum sequence length from training sequences
    min_length = None
    max_length = None
    for seq_id, sequence in sequences.items():
        seq_len = len(sequence)
        if min_length is None or seq_len < min_length:
            min_length = seq_len
        if max_length is None or seq_len > max_length:
            max_length = seq_len
    return min_length, max_length

def detect_length(min_length, max_length, sequences):
    # Identify all sequences with length lower than minimum or higher than maximum of training sequence lengths
    detected = set()
    for seq_id, sequence in sequences.items():
        if len(sequence) < min_length or len(sequence) > max_length:
            detected.add(seq_id)
    return detected

def train_ngram(n, sequences):
    # Get all n-grams from training sequences
    ngram_model = {}
    for seq_id, seq in sequences.items():
        seq = [-1] + seq + [-1]
        if n not in ngram_model:
             ngram_model[n] = set()
        for i in range(len(seq) - n + 1):
            ngram = tuple(seq[i:(i+n)])
            ngram_model[n].add(ngram)
    return ngram_model

def train_ngram_cv(n, sequences, idf):
    # Get all n-gram count vectors from training sequences
    train_vectors = {n: []}
    idf_weights = {}
    cnt = 0
    for seq_id, seq in sequences.items():
        cnt += 1
        ngram_model = {}
        seq = [-1] + seq + [-1]
        for i in range(len(seq) - n + 1):
            ngram = tuple(seq[i:(i+n)])
            if ngram not in ngram_model:
                ngram_model[ngram] = 1
            else:
                ngram_model[ngram] += 1
            if ngram not in idf_weights:
                idf_weights[ngram] = set(seq_id)
            else:
                idf_weights[ngram].add(seq_id)
        train_vectors[n].append(ngram_model)
    N = cnt
    for event_type in idf_weights:
        idf_weights[event_type] = math.log10((1 + N) / len(idf_weights[event_type]))
    return train_vectors, idf_weights

def get_ngram_cv_vectors(n, sequences):
    # Get all n-gram count vectors from test sequences
    test_vectors = {}
    for seq_id, seq in sequences.items():
        seq = [-1] + seq + [-1]
        if n not in test_vectors:
            test_vectors[n] = {}
        if seq_id not in test_vectors[n]:
            test_vectors[n][seq_id] = {}
        for i in range(len(seq) - n + 1):
            ngram = tuple(seq[i:(i+n)])
            if ngram not in test_vectors[n]:
                test_vectors[n][seq_id][ngram] = 1
            else:
                test_vectors[n][seq_id][ngram] += 1
    return test_vectors

def detect_ngram(ngram_model, n, sequences):
    # Detection by n-grams following the STIDE approach
    # See Forrest, Stephanie, et al. "A sense of self for unix processes." Proceedings 1996 IEEE Symposium on Security and Privacy. IEEE, 1996
    detected = {}
    mn_max = 0
    for seq_id, seq in sequences.items():
        m = 0
        seq = [-1] + seq + [-1]
        for i in range(len(seq) - n + 1):
            ngram = tuple(seq[i:(i+n)])
            if ngram not in ngram_model[n]:
                m += 1
                break
        if n >= len(seq):
            mmax = 1
        else:
            mmax = n * (len(seq) - n / 2)
        mn = m / mmax
        detected[seq_id] = mn
        mn_max = max(mn_max, mn)
    # Scale in [0, 1]
    for seq_id, mn in detected.items():
        if mn_max == 0:
            detected[seq_id] = 0
        else:
            detected[seq_id] = mn / mn_max
    return iterate_threshold(detected)

def detect_ngram_old(ngram_model, n, sequences):
    # This method has been replaced by detect_ngram
    # Detection by n-grams (single unknown n-gram means that whole sequence is detected)
    detected = set()
    for seq_id, seq in sequences.items():
        seq = [-1] + seq + [-1]
        for i in range(len(seq) - n + 1):
            ngram = tuple(seq[i:(i+n)])
            if ngram not in ngram_model[n]:
                detected.add(seq_id)
                break
    return detected

def get_event_times(lines, tw):
    # Compute event inter-arrival times
    event_times = {}
    seq_prev = {}
    all_seq_ids = set()
    header = True
    for line in lines:
        if header:
            header = False
            colnames = line.strip('\n').split(';')
            continue
        parts = line.strip('\n').split(';')
        event_id = parts[colnames.index('event_type')]
        time = float(parts[colnames.index('time')])
        if tw is not None:
            seq_id = str(int(math.floor(time / tw) * tw)) + '.0'
        else:
            seq_id = parts[colnames.index('seq_id')]
        all_seq_ids.add(seq_id)
        if seq_id not in seq_prev:
            seq_prev[seq_id] = (event_id, time)
            continue
        prev_event_id, prev_event_time = seq_prev[seq_id]
        event_pair = (prev_event_id, event_id)
        interarrival_time = time - prev_event_time
        if seq_id not in event_times:
            event_times[seq_id] = {}
        if event_pair not in event_times[seq_id]:
            event_times[seq_id][event_pair] = (interarrival_time, interarrival_time)
        else:
            event_times[seq_id][event_pair] = (min(interarrival_time, event_times[seq_id][event_pair][0]), max(interarrival_time, event_times[seq_id][event_pair][1]))
        seq_prev[seq_id] = (event_id, time)
    return event_times, all_seq_ids

import re
def get_event_params(lines):
    # Get all event parameters by position in event type
    event_params = {}
    header = True
    regex = re.compile('[^a-zA-Z]')
    for line in lines:
        if header:
            header = False
            colnames = line.strip('\n').split(';')
            continue
        parts = line.strip('\n').split(';')
        seq_id = parts[colnames.index('seq_id')]
        event_id = parts[colnames.index('event_type')]
        params = parts[colnames.index('params')].split('§')
        if seq_id not in event_params:
            event_params[seq_id] = {}
        if event_id not in event_params[seq_id]:
            event_params[seq_id][event_id] = {}
        for i, param_raw in enumerate(params):
            # Store parameter position within event type in i
            param = regex.sub('', param_raw) # Replace special characters and digits
            if i not in event_params[seq_id][event_id]:
                event_params[seq_id][event_id][i] = set([param])
            else:
                event_params[seq_id][event_id][i].add(param)
    return event_params

def get_word_dict(event_params, sequences):
    # Learn all parameter values that occur in specific positions of event types
    word_dict = {}
    for seq_id in sequences:
        for event_id, pos_d in event_params[seq_id].items():
            if event_id not in word_dict:
                word_dict[event_id] = {}
            for pos, params in pos_d.items():
                if pos not in word_dict[event_id]:
                    word_dict[event_id][pos] = params
                else:
                    word_dict[event_id][pos].update(params)
    return word_dict

import pylcs
def detect_word_dict(word_dict, event_params, sequences):
    # Identify all sequences where new values occur in parameters that had only few values in training sequences
    detected = set()
    for seq_id in sequences:
        found = False
        for event_id, pos_d in event_params[seq_id].items():
            for pos, params in pos_d.items():
                if event_id not in word_dict:
                    continue
                if pos not in word_dict[event_id]:
                    continue
                # Report as anomaly if this token position should be static or categorical (i.e., only have few values) and observed value deviates from that
                if len(word_dict[event_id][pos]) < 3 and len(params.difference(word_dict[event_id][pos])) > 0:
                    detected.add(seq_id)
                    found = True
                    break
            if found:
                break
    return detected

import copy
def find_time_range_rec(start, goal, time_ranges, path, depth):
    # Recursive function to find inter-arrival times of event pairs with some additional events in between
    results = []
    if depth > 1:
        return []
    for ep in time_ranges:
        if ep[0] == start and ep[1] == goal:
            return [path + [ep[0], ep[1]]]
        if ep[0] == start:
            time_ranges_tmp = copy.deepcopy(time_ranges)
            del time_ranges_tmp[ep]
            for rec_result in find_time_range_rec(ep[1], goal, time_ranges_tmp, path + [ep[0]], depth + 1):
                results.append(rec_result)
    return results

def update_time_range(event_pair, time_ranges):
    # This function will adapt the inter-arrival time of event pairs with some additional events in between
    results = find_time_range_rec(event_pair[0], event_pair[1], time_ranges, [], 0)
    min_min = None
    max_max = None
    for result in results:
        prev = None
        min_sum = 0
        max_sum = 0
        for elem in result:
            if prev is None:
                prev = elem
                continue
            min_sum += time_ranges[(prev, elem)]["min"]
            max_sum += time_ranges[(prev, elem)]["max"]
            prev = elem
        if min_min is None:
            min_min = min_sum
        else:
            min_min = min(min_min, min_sum)
        if max_max is None:
            max_max = max_sum
        else:
            max_max = max(max_max, max_sum)
    if min_min is not None:
        time_ranges[event_pair] = {"min": min_min, "max": max_max}
    else:
        time_ranges[event_pair] = {"min": -1, "max": -1}

import scipy
from scipy import stats
def detect_event_times_range(time_ranges, event_times, sequences):
    # Identify all sequences where event pairs have deviating (i.e., too long or too short) inter-arrival times
    detected_score = {}
    for seq_id in sequences:
        detected_score[seq_id] = []
        if seq_id not in event_times:
            # No interarrival time exists when seq_id has only a single event
            continue
        for event_pair, times_list in event_times[seq_id].items():
            if event_pair not in time_ranges:
                continue
                #update_time_range(event_pair, time_ranges)
            #if time_ranges[event_pair]["min"] == -1:
            #    # Set to -1 when known from previous iterations that event pair cannot be estimated
            #    continue
            if True:
                score_min = 0
                score_max = 0
                if min(times_list) < time_ranges[event_pair]["min"]:
                    if time_ranges[event_pair]["min"] != 0:
                        score_min = (time_ranges[event_pair]["min"] - min(times_list)) / time_ranges[event_pair]["min"]
                    else:
                        score_min = 1
                    detected_score[seq_id].append(score_min)
                if max(times_list) > time_ranges[event_pair]["max"]:
                    if max(times_list) != 0:
                        score_max = (max(times_list) - time_ranges[event_pair]["max"]) / max(times_list)
                    else:
                        score_max = 1
                    detected_score[seq_id].append(score_max)
            # Alternatively, mean and std could be used, but problematic when only few (or one) observations exist for event pair
            #else:
            #    for time in times_list:
            #        v = scipy.stats.norm.pdf(time, loc=time_ranges[event_pair]["mean"], scale=time_ranges[event_pair]["std"])
    detected_dict = {}
    for i in range(100):
        detected = set()
        threshold = i / 100.0
        for seq_id, score_list in detected_score.items():
            if len(score_list) > 0 and max(score_list) > threshold:
                detected.add(seq_id)
        detected_dict[threshold] = detected
    return detected_dict

import numpy as np
def get_event_times_range(event_times, sequences):
    # Get minimum and maximum as well as mean and standard deviation from event pair inter-arrival times
    event_times_agg = {}
    for seq_id in sequences:
        if seq_id not in event_times:
            # No interarrival time exists when seq_id has only a single event
            continue
        for event_pair, times_list in event_times[seq_id].items():
            if event_pair not in event_times_agg:
                event_times_agg[event_pair] = times_list
            else:
                #event_times_agg[event_pair].extend(times_list)
                event_times_agg[event_pair] = (min(times_list[0], event_times_agg[event_pair][0]), max(times_list[1], event_times_agg[event_pair][1]))
    event_times_range = {}
    for event_pair, times_list in event_times_agg.items():
        if event_pair in event_times_range:
            print('Warning: Overwriting event pair')
        # Todo: Implement a rolling mean and std
        #event_times_range[event_pair] = {"min": min(times_list), "max": max(times_list), "mean": np.mean(times_list), "std": np.std(times_list)}
        event_times_range[event_pair] = {"min": times_list[0], "max": times_list[1]}
    return event_times_range

def evaluate_all(data_dir, time_det, normalize, tw=None):
    train_sequences = None
    test_normal_sequences = None
    test_abnormal_sequences = None
    # Load training and test sequences
    with open(data_dir + '/' + data_dir.split('_')[0] + '_train') as train, open(data_dir + '/' + data_dir.split('_')[0] + '_test_abnormal') as test_abnormal, open(data_dir + '/' + data_dir.split('_')[0] + '_test_normal') as test_normal:
        train_sequences = read_lines(train)
        test_normal_sequences = read_lines(test_normal)
        test_abnormal_sequences = read_lines(test_abnormal)
    if time_det:
        # Only load timing and parameter information when parsed logs are available
        with open(data_dir + '/parsed.csv') as parsed:
            event_times, parsed_seq_ids = get_event_times(parsed, tw) # Required for detection based on event inter-arrival times
            vector_seq_ids = set(list(train_sequences.keys()) + list(test_normal_sequences.keys()) + list(test_abnormal_sequences.keys()))
            if parsed_seq_ids != vector_seq_ids:
                print('WARNING: Mismatching sequence IDS in parsed (' + str(len(parsed_seq_ids)) + ' sequences) and vector files (' + str(len(vector_seq_ids)) + ' sequences). Did you use the correct time_window?')
        #with open(data_dir + '/parsed.csv') as parsed:
        #    event_params = get_event_params(parsed) # Required for detection based on parameter values
    if len(set(test_normal_sequences.keys()).intersection(test_abnormal_sequences.keys())) != 0:
        print('WARNING: Same sequence IDs occur in normal and abnormal test sets: ' + str(test_normal_sequences.keys()).intersection(test_abnormal_sequences.keys()))
    test_sequences = {**test_abnormal_sequences, **test_normal_sequences} # Merge test sequences

    results = []
    # Detection based on new events
    known_event_types = get_known_event_types(train_sequences)
    start_time = time.time()
    detected_known_event_types = detect_known_event_types(known_event_types, test_sequences)
    new_event_time = time.time() - start_time
    tp, fn, tn, fp = evaluate(detected_known_event_types, test_normal_sequences.keys(), test_abnormal_sequences.keys())
    results.append(print_results('New event detection', tp, fn, tn, fp, None, new_event_time))
    # Detection based on sequence lengths
    min_length, max_length = get_length_range(train_sequences)
    start_time = time.time()
    detected_length = detect_length(min_length, max_length, test_sequences)
    length_time = time.time() - start_time
    tp, fn, tn, fp = evaluate(detected_length, test_normal_sequences.keys(), test_abnormal_sequences.keys())
    results.append(print_results('Sequence length detection', tp, fn, tn, fp, None, length_time))
    # Combination new events and length
    detected_comb_events_length = detected_known_event_types.union(detected_length)
    tp, fn, tn, fp = evaluate(detected_comb_events_length, test_normal_sequences.keys(), test_abnormal_sequences.keys())
    results.append(print_results('New events + sequence length detection', tp, fn, tn, fp, None, new_event_time + length_time))
    # Count vector clustering
    idf = False
    test_vectors = get_vectors(test_sequences)
    train_vectors, known_event_types, idf_weights = train_cluster_count_vectors(train_sequences, idf)
    start_time = time.time()
    detected_dict = test_cluster_count_vectors(train_vectors, test_vectors, test_normal_sequences.keys(), test_abnormal_sequences.keys(), normalize, idf, idf_weights)
    cvc_time = time.time() - start_time
    best_threshold = get_best(detected_dict, set(), test_normal_sequences.keys(), test_abnormal_sequences.keys())
    tp, fn, tn, fp = evaluate(detected_dict[best_threshold], test_normal_sequences.keys(), test_abnormal_sequences.keys())
    results.append(print_results('Count vector clustering', tp, fn, tn, fp, best_threshold, cvc_time))
    # Count vector clustering with idf
    idf = True
    test_vectors = get_vectors(test_sequences)
    train_vectors, known_event_types, idf_weights = train_cluster_count_vectors(train_sequences, idf)
    start_time = time.time()
    detected_dict_idf = test_cluster_count_vectors(train_vectors, test_vectors, test_normal_sequences.keys(), test_abnormal_sequences.keys(), normalize, idf, idf_weights)
    cvc_idf_time = time.time() - start_time
    best_threshold_idf = get_best(detected_dict_idf, set(), test_normal_sequences.keys(), test_abnormal_sequences.keys())
    tp, fn, tn, fp = evaluate(detected_dict_idf[best_threshold_idf], test_normal_sequences.keys(), test_abnormal_sequences.keys())
    results.append(print_results('Count vector clustering with idf', tp, fn, tn, fp, best_threshold_idf, cvc_idf_time))
    # Detection based on event interval times
    if time_det:
        # Only possible when parsed event information is available
        time_ranges = get_event_times_range(event_times, train_sequences)
        start_time = time.time()
        detected_times_range_dict = detect_event_times_range(time_ranges, event_times, test_sequences)
        new_event_time = time.time() - start_time
        best_threshold = get_best(detected_times_range_dict, set(), test_normal_sequences.keys(), test_abnormal_sequences.keys())
        tp, fn, tn, fp = evaluate(detected_times_range_dict[best_threshold], test_normal_sequences.keys(), test_abnormal_sequences.keys())
        results.append(print_results('Time interval detection', tp, fn, tn, fp, best_threshold, new_event_time))
    # 2-gram
    twogram_model = train_ngram(2, train_sequences)
    start_time = time.time()
    detected_twogram = detect_ngram(twogram_model, 2, test_sequences)
    twogram_time = time.time() - start_time
    best_threshold = get_best(detected_twogram, set(), test_normal_sequences.keys(), test_abnormal_sequences.keys())
    tp, fn, tn, fp = evaluate(detected_twogram[best_threshold], test_normal_sequences.keys(), test_abnormal_sequences.keys())
    results.append(print_results('2-gram detection', tp, fn, tn, fp, best_threshold, twogram_time))
    best_threshold_twogram = best_threshold
    # 3-gram
    threegram_model = train_ngram(3, train_sequences)
    start_time = time.time()
    detected_threegram = detect_ngram(threegram_model, 3, test_sequences)
    threegram_time = time.time() - start_time
    best_threshold = get_best(detected_threegram, set(), test_normal_sequences.keys(), test_abnormal_sequences.keys())
    tp, fn, tn, fp = evaluate(detected_threegram[best_threshold], test_normal_sequences.keys(), test_abnormal_sequences.keys())
    results.append(print_results('3-gram detection', tp, fn, tn, fp, best_threshold, threegram_time))
    # 10-gram
    tengram_model = train_ngram(10, train_sequences)
    start_time = time.time()
    detected_tengram = detect_ngram(tengram_model, 10, test_sequences)
    tengram_time = time.time() - start_time
    best_threshold = get_best(detected_tengram, set(), test_normal_sequences.keys(), test_abnormal_sequences.keys())
    tp, fn, tn, fp = evaluate(detected_tengram[best_threshold], test_normal_sequences.keys(), test_abnormal_sequences.keys())
    results.append(print_results('10-gram detection', tp, fn, tn, fp, best_threshold, tengram_time))
    # Combination of 2-gram and sequence length detection
    detected_twogram_len = detected_twogram[best_threshold_twogram].union(detected_comb_events_length)
    tp, fn, tn, fp = evaluate(detected_twogram_len, test_normal_sequences.keys(), test_abnormal_sequences.keys())
    results.append(print_results('2-gram + sequence length detection', tp, fn, tn, fp, best_threshold_twogram, twogram_time + length_time))
    # Combination new events, length, and count vector clustering
    best_threshold = get_best(detected_dict, detected_comb_events_length, test_normal_sequences.keys(), test_abnormal_sequences.keys())
    detected_comb_ev_len_ccv = detected_dict[best_threshold].union(detected_comb_events_length)
    tp, fn, tn, fp = evaluate(detected_comb_ev_len_ccv, test_normal_sequences.keys(), test_abnormal_sequences.keys())
    results.append(print_results('New events + sequence length detection + count vector clustering', tp, fn, tn, fp, best_threshold, new_event_time + length_time + cvc_time))
    # Combination new events, length, and count vector clustering with idf
    best_threshold_idf = get_best(detected_dict_idf, detected_comb_events_length, test_normal_sequences.keys(), test_abnormal_sequences.keys())
    detected_comb_ev_len_ccv_idf = detected_dict_idf[best_threshold_idf].union(detected_comb_events_length)
    tp, fn, tn, fp = evaluate(detected_comb_ev_len_ccv_idf, test_normal_sequences.keys(), test_abnormal_sequences.keys())
    results.append(print_results('New events + sequence length detection + count vector clustering with idf', tp, fn, tn, fp, best_threshold_idf, new_event_time + length_time + cvc_idf_time))
    # Detection based on edit distance
    start_time = time.time()
    detected_edit_dict = test_edit_distance(train_sequences, test_sequences)
    edit_time = time.time() - start_time
    best_threshold = get_best(detected_edit_dict, set(), test_normal_sequences.keys(), test_abnormal_sequences.keys())
    tp, fn, tn, fp = evaluate(detected_edit_dict[best_threshold], test_normal_sequences.keys(), test_abnormal_sequences.keys())
    results.append(print_results('Edit distance detection', tp, fn, tn, fp, best_threshold, edit_time))
    # Combination of new events + length + edit
    best_threshold = get_best(detected_edit_dict, detected_comb_events_length, test_normal_sequences.keys(), test_abnormal_sequences.keys())
    detected_comb_ev_len_edit = detected_edit_dict[best_threshold].union(detected_comb_events_length)
    tp, fn, tn, fp = evaluate(detected_comb_ev_len_edit, test_normal_sequences.keys(), test_abnormal_sequences.keys())
    results.append(print_results('New events + sequence length detection + edit distance', tp, fn, tn, fp, best_threshold, new_event_time + length_time + edit_time))
    # Detection based on parameter values
    if False: # This detection method needs refinement
        # This requires that parsed.csv is available and that parameters have been parsed (e.g., python3 hadoop_parse.py --output_params True)
        word_dict = get_word_dict(event_params, train_sequences)
        start_time = time.time()
        detected_word_dict = detect_word_dict(word_dict, event_params, test_sequences)
        word_dict_time = time.time() - start_time
        tp, fn, tn, fp = evaluate(detected_word_dict, test_normal_sequences.keys(), test_abnormal_sequences.keys())
        results.append(print_results('Token-based detection', tp, fn, tn, fp, None, word_dict_time))
    return results

if __name__ == "__main__":
    evaluate_all(data_dir, time_det, True, tw)
