import datetime
from datetime import timezone
from glob import glob
import os
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--data_dir", default="adfa_verazuo", help="path to input files", type=str, choices=['adfa_verazuo'])
parser.add_argument("--sep_csv", default=";", help="separator for values used in output file", type=str)

params = vars(parser.parse_args())
source = params["data_dir"]
sep_csv = params["sep_csv"]

logfiles = [y for x in os.walk(source + '/ADFA-LD') for y in glob(os.path.join(x[0], '*.txt'))]

with open(source + '/parsed.csv', 'w+') as ext_file:
    ext_file.write('id;event_type;seq_id;time;label\n')
    cnt = 0
    for logfile in logfiles:
        if 'ADFA-LD+Syscall+List.txt' in logfile:
            # Skip label file
            continue
        with open(logfile) as logsource:
            logfile_parts = logfile.strip('\n').split('/')
            if '/Attack_Data_Master/' in logfile:
                label = '_'.join(logfile_parts[3].split('_')[:-1])
            else:
                label = 'Normal'
            seq_id = logfile_parts[-1].replace('.txt', '') # Use file name as sequence identifier
            for line in logsource:
                for event_id in line.strip('\n ').split(' '):
                    cnt += 1
                    if cnt % 100000 == 0:
                        print(str(cnt) + ' events processed')
                    csv_line = str(cnt) + sep_csv + str(event_id) + sep_csv + str(seq_id) + sep_csv + '-1' + sep_csv + str(label)
                    ext_file.write(csv_line + '\n')
