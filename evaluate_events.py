import random
import argparse
import time

parser = argparse.ArgumentParser()

parser.add_argument("--data_dir", default="bgl_cfdr", help="path to input files", type=str, choices=['bgl_loghub', 'bgl_cfdr', 'thunderbird_cfdr'])
parser.add_argument("--train_ratio", default=0.01, help="fraction of normal data used for training", type=float)
parser.add_argument("--sorting", default="random", help="sorting mode", type=str, choices=['random', 'chronological'])

params = vars(parser.parse_args())
source = params["data_dir"]
train_ratio = params["train_ratio"]
sorting = params["sorting"]

def get_fone(tp, fn, tn, fp):
    if tp + fp + fn == 0:
        return 'inf'
    return tp / (tp + 0.5 * (fp + fn))

def print_results(name, tp, fn, tn, fp, det_time):
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
    print('')
    print(name)
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
    return {'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn, 'tpr': tpr, 'fpr': fpr, 'tnr': tnr, 'p': p, 'f1': fone, 'acc': acc, 'name': name, 'time': det_time}

def evaluate_all(source):
    normal_events = []
    abnormal_events = []
    header = True
    with open(source.split('/')[0] + '/parsed.csv') as extracted:
        print('Read in parsed events ...')
        for line in extracted:
            if header:
                header = False
                colnames = line.strip('\n').split(';')
                continue
            parts = line.strip('\n').split(';')
            event_id = parts[colnames.index('event_type')]
            label = parts[colnames.index('eventlabel')]
            if label == "Normal":
                normal_events.append(event_id)
            else:
                abnormal_events.append(event_id)
        num_train_logs = int(train_ratio * len(normal_events))
        if sorting == "random":
            print('Randomly selecting ' + str(num_train_logs) + ' events from ' + str(len(normal_events)) + ' normal events for training')
            random.shuffle(normal_events)
        elif sorting == "chronological":
            print('Chronologically selecting ' + str(num_train_logs) + ' events from ' + str(len(normal_events)) + ' normal events for training')
            pass
        else:
            print("Warning: Unknown sorting mode!")
        train = set(normal_events[:num_train_logs])
        test_normal = normal_events[num_train_logs:]
        known_events = set(train)
        tp = 0
        fp = 0
        fn = 0
        tn = 0
        start_time = time.time()
        print('Testing ' + str(len(test_normal)) + ' normal events')
        for elem in test_normal:
            if elem in known_events:
                tn += 1
            else:
                fp += 1
        print('Testing ' + str(len(abnormal_events)) + ' anomalous events')
        for elem in abnormal_events:
            if elem in known_events:
                fn += 1
            else:
                tp += 1
        return print_results("New events", tp, fn, tn, fp, time.time() - start_time)

if __name__ == "__main__":
    evaluate_all(source)
