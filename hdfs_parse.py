import datetime
from datetime import timezone
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--data_dir", default="hdfs_xu", help="path to input files", type=str, choices=['hdfs_xu', 'hdfs_loghub'])
parser.add_argument("--output_line", default="False", help="original line is provided in resulting csv", type=str)
parser.add_argument("--output_params", default="False", help="extracted params are provided in resulting csv", type=str)
parser.add_argument("--early_stopping", default="True", help="setting this to True will use the first matching template; if False, all templates will be checked (e.g. for lines that match multiple templates)", type=str)
parser.add_argument("--sep_csv", default=";", help="separator for values used in output file", type=str)
parser.add_argument("--sep_params", default="§", help="separator for params (if --output_params is set to True) used in output file", type=str)

params = vars(parser.parse_args())
source = params["data_dir"]
output_line = params["output_line"] == "True"
output_params = params["output_params"] == "True"
early_stopping = params["early_stopping"] == "True"
sep_csv = params["sep_csv"]
sep_params = params["sep_params"]

templates = []
labels = {}

hdfs_file = ""
if source == "hdfs_xu":
    hdfs_file = "sorted.log"
elif source == "hdfs_loghub":
    hdfs_file = "HDFS.log"
else:
    print('Unknown source ' + str(source))
    exit()

with open('hdfs_loghub/anomaly_label.csv') as labels_file:
    first = True
    normal_logs = []
    for line in labels_file:
        if first:
            first = False
            continue
        parts = line.split(',')
        name = parts[0]
        label = parts[1].strip('\n')
        labels[name] = label

with open(source + '/' + hdfs_file) as log_file, open('templates/HDFS_templates.csv') as templates_file, open(source + '/parsed.csv', 'w+') as ext_file:
    header = 'id' + sep_csv + 'event_type' + sep_csv + 'seq_id' + sep_csv + 'time' + sep_csv + 'label'
    if output_line:
        header += sep_csv + "line"
    if output_params:
        header += sep_csv + "params"
    ext_file.write(header + '\n')
    for line in templates_file:
        template = line.strip('\n').rstrip(' ').strip('<*>').split('<*>')
        templates.append(template)
    cnt = 0
    prev_timestamp = None
    for line in log_file:
        cnt += 1
        if cnt % 100000 == 0:
            print(str(cnt) + ' lines processed')
        line = line.strip('\n')
        template_id = None
        seq_ids = []
        found_params = []
        for i, template in enumerate(templates):
            matches = []
            params = []
            cur = 0
            for template_part in template:
                pos = line.find(template_part, cur)
                if pos == -1:
                    matches = [] # Reset matches so that it counts as no match at all
                    break
                matches.append(pos)
                if line[cur:pos] != '':
                    params.append(line[cur:pos])
                cur = pos + len(template_part)
            if len(matches) > 0 and sorted(matches) == matches:
                if template_id is not None:
                    print('WARNING: Templates ' + str(template_id) + ' and ' + str(i + 1) + ' both match line ' + line)
                template_id = i + 1 # offset by 1 so that ID matches with template file
                if line[cur:] != '':
                    params.append(line[cur:])
                found_params = params # Store params found for matching template since params variable will be reset when checking next template
                if template_id == 30: # Contains multiple block ids
                    seq_ids = params[2].strip(' ').split(' ')
                else:
                    for param in params:
                        pos = param.find('blk_')
                        if pos != -1:
                            seq_id_found = param[pos:].split(' ')[0].split("'")[0].split('.')[0] # blk_<id> is followed either by space, quote, or dot
                            if len(seq_ids) != 0 and seq_id_found != seq_ids[0]: # Sometimes same blk_<id> occurs multiple times in same line
                                print('WARNING: Multiple sequence IDs in line ' + str(line))
                            if param.find('blk_', pos + 1) != -1:
                                print('WARNING: Multiple sequence IDs in line ' + str(line))
                            seq_ids = [seq_id_found]
                if early_stopping:
                    break
        if template_id is None:
            print('WARNING: No template matches ' + str(line))
        if len(seq_ids) == 0:
            print('WARNING: No sequence id found in line ' + str(line))
        time_string = line[:13] # timestamp format in logs: 081111 111607
        if time_string == "du: cannot ac": # last few lines in log file do not have a time stamp
            timestamp = prev_timestamp # assume that the lines without timestamp occur at the same time as the logs before
        else:
            timestamp = datetime.datetime(year=int('20' + time_string[:2]), month=int(time_string[2:4]), day=int(time_string[4:6]), hour=int(time_string[7:9]), minute=int(time_string[9:11]), second=int(time_string[11:13])).replace(tzinfo=timezone.utc).timestamp()
            prev_timestamp = timestamp
        for seq_id in seq_ids:
            csv_line = str(cnt) + sep_csv + str(template_id) + sep_csv + str(seq_id) + sep_csv + str(timestamp) + sep_csv + str(labels[seq_id])
            if output_line:
                csv_line += sep_csv + str(line)
            if output_params:
                csv_line += sep_csv + sep_params.join(found_params)
            ext_file.write(csv_line + '\n')
