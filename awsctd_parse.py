import datetime
from datetime import timezone
from glob import glob
import os
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--data_dir", default="awsctd_djpasco", help="path to input files", type=str, choices=['awsctd_djpasco'])
parser.add_argument("--sep_csv", default=";", help="separator for values used in output file", type=str)

params = vars(parser.parse_args())
source = params["data_dir"]
sep_csv = params["sep_csv"]

logfiles = [y for x in os.walk(source + '/CSV') for y in glob(os.path.join(x[0], '*.csv'))]

with open(source + '/parsed.csv', 'w+') as ext_file:
    ext_file.write('id' + sep_csv + 'event_type' + sep_csv + 'seq_id' + sep_csv + 'time' + sep_csv + 'label\n')
    cnt = 0
    for logfile in logfiles:
        with open(logfile) as logsource:
            logfile_parts = logfile.strip('\n').split('/')
            line_nr = 1
            for line in logsource:
                line_parts = line.strip('\n').split(',')
                label = line_parts[-1]
                if label == "Clean":
                    label = "Normal"
                seq_id = logfile_parts[-2] + '/' + logfile_parts[-1].replace('.csv', '') + '_' + str(line_nr) # Use filename + incrementing id per sequence
                for event_id in line_parts[:-1]:
                    cnt += 1
                    if cnt % 1000000 == 0:
                        print(str(cnt) + ' events processed')
                    csv_line = str(cnt) + sep_csv + str(event_id) + sep_csv + str(seq_id) + sep_csv + '-1' + sep_csv + str(label)
                    ext_file.write(csv_line + '\n')
                line_nr += 1
