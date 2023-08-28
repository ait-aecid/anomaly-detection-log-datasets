import datetime
from datetime import timezone
from glob import glob
import os
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--data_dir", default="hadoop_loghub", help="path to input files", type=str, choices=['hadoop_loghub'])
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

logfiles = [y for x in os.walk(source + '/logs') for y in glob(os.path.join(x[0], '*.log'))]

with open('templates/Hadoop_templates.csv') as templates_file:
    for line in templates_file:
        template = line.strip('\n').rstrip(' ').split('<*>')
        templates.append(template)

with open(source + '/parsed.csv', 'w+') as ext_file:
    header = 'id' + sep_csv + 'event_type' + sep_csv + 'seq_id' + sep_csv + 'time' + sep_csv + 'label'
    if output_line:
        header += sep_csv + "line"
    if output_params:
        header += sep_csv + "params"
    ext_file.write(header + '\n')
    cnt = 0
    for logfile in logfiles:
        prev_timestamp = None
        with open(logfile) as logsource:
            logfile_parts = logfile.split('/')
            app_id = logfile_parts[-2]
            label = None
            if app_id in ['application_1445087491445_0005', 'application_1445087491445_0007', 'application_1445175094696_0005'] + ['application_1445062781478_0011', 'application_1445062781478_0016', 'application_1445062781478_0019', 'application_1445076437777_0002', 'application_1445076437777_0005', 'application_1445144423722_0021', 'application_1445144423722_0024', 'application_1445182159119_0012']:
                label = 'Normal'
            elif app_id in ['application_1445087491445_0001', 'application_1445087491445_0002', 'application_1445087491445_0003', 'application_1445087491445_0004', 'application_1445087491445_0006', 'application_1445087491445_0008', 'application_1445087491445_0009', 'application_1445087491445_0010', 'application_1445094324383_0001', 'application_1445094324383_0002', 'application_1445094324383_0003', 'application_1445094324383_0004', 'application_1445094324383_0005'] + ['application_1445062781478_0012', 'application_1445062781478_0013', 'application_1445062781478_0014', 'application_1445062781478_0015', 'application_1445062781478_0017', 'application_1445062781478_0018', 'application_1445062781478_0020', 'application_1445076437777_0001', 'application_1445076437777_0003', 'application_1445076437777_0004', 'application_1445182159119_0016', 'application_1445182159119_0017', 'application_1445182159119_0018', 'application_1445182159119_0019', 'application_1445182159119_0020']:
                label = 'machine_down'
            elif app_id in ['application_1445175094696_0001', 'application_1445175094696_0002', 'application_1445175094696_0003', 'application_1445175094696_0004'] + ['application_1445144423722_0020', 'application_1445144423722_0022', 'application_1445144423722_0023']:
                label = 'network_disconnection'
            elif app_id in ['application_1445182159119_0001', 'application_1445182159119_0002', 'application_1445182159119_0003', 'application_1445182159119_0004', 'application_1445182159119_0005'] + ['application_1445182159119_0011', 'application_1445182159119_0013', 'application_1445182159119_0014', 'application_1445182159119_0015']:
                label = 'disk_full'
            else:
                print('WARNING: application id ' + str(app_id) + ' is unknown')
            container_id = logfile_parts[-1].replace('.log', '')
            for line in logsource:
                cnt += 1
                if cnt % 100000 == 0:
                    print(str(cnt) + ' lines processed')
                line = line.replace('\t', ' ').rstrip('\n')
                template_id = None
                line_parts = line.split(' ')
                line_content_start = line.find(': ')
                timestamp_missing = False
                if line.startswith("2015-10"):
                    line = line[(line_content_start + 2):]
                else:
                    timestamp_missing = True
                if line == "":
                    continue
                found_params = []
                for i, template in enumerate(templates):
                    matches = []
                    params = []
                    cur = 0
                    for template_part in template:
                        pos = line.find(template_part, cur)
                        if pos == -1 or (' ' in line[cur:pos] and i not in [114, 115]):
                            matches = [] # Reset matches so that it counts as no match at all
                            break
                        matches.append(pos)
                        if line[cur:pos] != '':
                            params.append(line[cur:pos])
                        cur = pos + len(template_part)
                    if len(matches) > 0 and sorted(matches) == matches and (i not in [247] or line[cur:] == ' ') and (i not in [293] or line[cur:] == ''): # In E248, E293 make sure that nothing follows colon as in other related event types
                        if template_id is not None:
                            print('WARNING: Templates ' + str(template_id) + ' and ' + str(i + 1) + ' both match line ' + line)
                        template_id = i + 1 # offset by 1 so that ID matches with template file
                        if line[cur:] != '':
                            params.append(line[cur:])
                        found_params = params # Store params found for matching template since params variable will be reset when checking next template
                        if early_stopping:
                            break
                if template_id is None:
                    print('WARNING: No template matches "' + str(line) + '"')
                    print(line_parts)
                try:
                    if timestamp_missing:
                        timestamp = prev_timestamp
                    else:
                        timestamp = datetime.datetime.strptime(line_parts[0] + ' ' + line_parts[1], '%Y-%m-%d %H:%M:%S,%f').replace(tzinfo=timezone.utc).timestamp() # timestamp format is 2015-10-17 18:16:56,078
                except:
                    print(line)
                prev_timestamp = timestamp
                csv_line = str(cnt) + sep_csv + str(template_id) + sep_csv + str(app_id) + '/' + str(container_id) + sep_csv + str(timestamp) + sep_csv + str(label)
                if output_line:
                    csv_line += sep_csv + str(line)
                if output_params:
                    csv_line += sep_csv + sep_params.join(found_params)
                ext_file.write(csv_line + '\n')
