import datetime
from datetime import timezone
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--data_dir", default="bgl_cfdr", help="path to input files", type=str, choices=['bgl_cfdr', 'bgl_loghub'])
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

bgl_file = ""
if source == "bgl_loghub":
    bgl_file = "BGL.log"
elif source == "bgl_cfdr":
    bgl_file = "bgl2"
else:
    print('Unknown source ' + str(source))
    exit()

print('Get labels ...')
anomalous_sequences = set()
with open(source + '/' + bgl_file) as log_file:
    for line in log_file:
        line = line.strip('\n ')
        line_parts = line.split(' ')
        label = line_parts[0]
        seq_id = line_parts[3]
        if label != '-':
            anomalous_sequences.add(seq_id)

events_allow_spaces = [82, 84, 172, 194, 293, 328, 362, 371, 397] # Line numbers in template file where <*> can represent multiple tokens separated by spaces.

print('Read lines ...')
with open(source + '/' + bgl_file) as log_file, open('templates/BGL_templates.csv') as templates_file, open(source + '/parsed.csv', 'w+') as ext_file:
    header = 'id' + sep_csv + 'event_type' + sep_csv + 'seq_id' + sep_csv + 'time' + sep_csv + 'label' + sep_csv + 'eventlabel'
    if output_line:
        header += sep_csv + "line"
    if output_params:
        header += sep_csv + "params"
    ext_file.write(header + '\n')
    for line in templates_file:
        template = line.strip('\n').rstrip(' ').split('<*>')
        templates.append(template)
    cnt = 0
    prev_timestamp = None
    for line in log_file:
        cnt += 1
        if cnt % 50000 == 0:
            print(str(cnt) + ' lines processed')
        line = line.strip('\n ')
        template_id = None
        line_parts = line.split(' ')
        seq_id = line_parts[3]
        eventlabel = line_parts[0]
        if eventlabel == '-':
            eventlabel = 'Normal'
        if seq_id in anomalous_sequences:
            label = "Anomaly"
        else:
            label = "Normal"
        found_params = []
        preamble = line_parts[:9]
        line = ' '.join(line_parts[9:])
        for i, template in enumerate(templates):
            if i == 382: # E384: Line consists of just a number
                if line.isdigit():
                    template_id = i + 1
                    found_params = [line]
                    break
                else:
                    continue
            if i == 383: # E385: Line is empty
                if line == '':
                    template_id = i + 1
                    found_params = ['']
                    break
                else:
                    continue
            matches = []
            params = []
            cur = 0
            starts_with_wildcard = False
            for template_part in template:
                if template_part == '' and cur == 0:
                    starts_with_wildcard = True
                pos = line.find(template_part, cur)
                if pos == -1 or (' ' in line[cur:pos] and i not in events_allow_spaces) or (not starts_with_wildcard and cur == 0 and pos != 0) or (i == 0 and not line.split(' ')[-1].split(':')[0].isdigit()):
                    matches = [] # Reset matches so that it counts as no match at all
                    break
                matches.append(pos)
                if line[cur:pos] != '':
                    params.append(line[cur:pos])
                cur = pos + len(template_part)
            if len(matches) > 0 and sorted(matches) == matches and (' ' not in line[cur:] or i in events_allow_spaces) and (line[cur:] == '' or template_part == ''):
                if template_id is not None:
                    print('WARNING: Templates ' + str(template_id) + ' and ' + str(i + 1) + ' both match line ' + line)
                template_id = i + 1 # offset by 1 so that ID matches with line in template file
                if line[cur:] != '':
                    params.append(line[cur:])
                found_params = params # Store params found for matching template since params variable will be reset when checking next template
                if early_stopping:
                    break
        if template_id is None:
            print('WARNING: No template matches ' + str(line))
        timestamp = datetime.datetime.strptime(line_parts[4], '%Y-%m-%d-%H.%M.%S.%f').replace(tzinfo=timezone.utc).timestamp()
        csv_line = str(cnt) + sep_csv + str(template_id) + sep_csv + str(seq_id) + sep_csv + str(timestamp) + sep_csv + str(label) + sep_csv + str(eventlabel)
        if output_line:
            csv_line += sep_csv + str(line)
        if output_params:
            csv_line += sep_csv + sep_params.join(found_params)
        ext_file.write(csv_line + '\n')
