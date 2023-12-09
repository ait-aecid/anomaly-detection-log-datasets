import datetime
from datetime import timezone
import os
import argparse
import re

parser = argparse.ArgumentParser()

parser.add_argument("--data_dir", default="openstack_loghub", help="path to input files", type=str, choices=['openstack_loghub', 'openstack_parisakalaki'])
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

normal_log_sources = []
abnormal_log_sources = []
templates = []

if source == 'openstack_parisakalaki':
    abnormal_log_sources = ['openstack-logs/openstack-vm-destroy-immediately-after-create.log', 'openstack-logs/openstack-nova-dhcpoff.log', 'openstack-logs/openstack-nova-undefine-vm-after-create.log']
    normal_log_sources = ['openstack-logs/openstack-nova-normal-vm-create.log']
    templates_source = 'templates/OpenStack_templates.csv'
else:
    abnormal_log_sources = ['openstack_abnormal.log']
    normal_log_sources = ['openstack_normal1.log', 'openstack_normal2.log']
    templates_source = 'templates/OpenStack_templates.csv'

log_abnormal_list = []
for abnormal_log_source in abnormal_log_sources:
    with open(source + '/' + abnormal_log_source) as filehandler:
        log_abnormal = []
        for line in filehandler:
            log_abnormal.append(line)
        log_abnormal_list.append(log_abnormal)
log_normal_list = []
for normal_log_source in normal_log_sources:
    with open(source + '/' + normal_log_source) as filehandler:
        log_normal = []
        for line in filehandler:
            log_normal.append(line)
        log_normal_list.append(log_normal)

check_missing_seq_ids = False
with open(templates_source) as templates_file, open(source + '/parsed.csv', 'w+') as ext_file:
    header = 'id' + sep_csv + 'event_type' + sep_csv + 'seq_id' + sep_csv + 'time' + sep_csv + 'label'
    if output_line:
        header += sep_csv + "line"
    if output_params:
        header += sep_csv + "params"
    ext_file.write(header + '\n')
    for line in templates_file:
        template = line.strip('\n').split('<*>')
        templates.append(template)
    skipped = 0
    cnt = 0
    for logsource in log_abnormal_list + log_normal_list:
        label = 0
        if logsource in log_abnormal_list:
            label = "Anomaly"
        else:
            label = "Normal"
        for line in logsource:
            cnt += 1
            if cnt % 10000 == 0:
                print(str(cnt) + ' lines processed')
            line = line.strip('\n')
            template_id = None
            # Get sequence id of form 00000000-0000-0000-0000-000000000000
            seq_pos = line.find('instance: ', 0)
            if seq_pos != -1:
                seq_id = line[seq_pos + len('instance: ') : seq_pos + len('instance: ') + len('00000000-0000-0000-0000-000000000000')]
                if line[seq_pos + len('instance: ') + len('00000000-0000-0000-0000-000000000000')] != ']':
                    print('WARNING: Instance ID not read correctly: ' + str(seq_id) + ' from ' + str(line))
            if seq_pos == -1:
                seq_pos = line.find(' for instance ', 0)
                if seq_pos != -1:
                    seq_id = line[seq_pos + len(' for instance ') : seq_pos + len(' for instance ') + len('00000000-0000-0000-0000-000000000000')]
                    if seq_pos + len(' for instance ') + len('00000000-0000-0000-0000-000000000000') != len(line) and line[seq_pos + len(' for instance ') + len('00000000-0000-0000-0000-000000000000')] != ':':
                        print('WARNING: Instance ID not read correctly: ' + str(seq_id) + ' from ' + str(line))
            if seq_pos == -1:
                seq_pos = line.find('/servers/', 0)
                if seq_pos != -1:
                    if line[seq_pos + len('/servers/') : seq_pos + len('/servers/') + len('detail')] == 'detail':
                        seq_pos = -1
                    else:
                        seq_id = line[seq_pos + len('/servers/') : seq_pos + len('/servers/') + len('00000000-0000-0000-0000-000000000000')]
                        if line[seq_pos + len('/servers/') + len('00000000-0000-0000-0000-000000000000')] != ' ':
                            print('WARNING: Instance ID not read correctly: ' + str(seq_id) + ' from ' + str(line))
            if seq_pos == -1:
                seq_pos = line.find(' domain id: ', 0)
                if seq_pos != -1:
                    seq_id = line[seq_pos + len(' domain id: ') : seq_pos + len(' domain id: ') + len('00000000-0000-0000-0000-000000000000')]
                    if line[seq_pos + len(' domain id: ') + len('00000000-0000-0000-0000-000000000000')] != ',':
                        print('WARNING: Instance ID not read correctly: ' + str(seq_id) + ' from ' + str(line))
            if seq_pos == -1:
                seq_pos = line.find('/instances/', 0)
                if seq_pos != -1:
                    if line[seq_pos + len('/instances/') : seq_pos + len('/instances/') + len('_base')] == '_base':
                        seq_pos = -1
                    else:
                        seq_id = line[seq_pos + len('/instances/') : seq_pos + len('/instances/') + len('00000000-0000-0000-0000-000000000000')]
                        if line[seq_pos + len('/instances/') + len('00000000-0000-0000-0000-000000000000')] != '/':
                            print('WARNING: Instance ID not read correctly: ' + str(seq_id) + ' from ' + str(line))
            if seq_pos == -1:
                seq_pos = line.find(' Instance ', 0)
                if seq_pos != -1:
                    seq_id = line[seq_pos + len(' Instance ') : seq_pos + len(' Instance ') + len('00000000-0000-0000-0000-000000000000')]
                    if line[seq_pos - 1] != ']' and line[seq_pos - 1] != ':':
                        print('WARNING: Instance ID not read correctly: ' + str(seq_id) + ' from ' + str(line))
            if seq_pos == -1:
                seq_pos = line.find(' Conflict updating instance ', 0)
                if seq_pos != -1:
                    seq_id = line[seq_pos + len(' Conflict updating instance ') : seq_pos + len(' Conflict updating instance ') + len('00000000-0000-0000-0000-000000000000')]
                    if line[seq_pos + len(' Conflict updating instance ') + len('00000000-0000-0000-0000-000000000000')] != '.':
                        print('WARNING: Instance ID not read correctly: ' + str(seq_id) + ' from ' + str(line))
            if seq_pos == -1:
                seq_pos = line.find(' consumer ', 0)
                if seq_pos != -1:
                    seq_id = line[seq_pos + len(' consumer ') : seq_pos + len(' consumer ') + len('00000000-0000-0000-0000-000000000000')]
                    if line[seq_pos + len(' consumer ') + len('00000000-0000-0000-0000-000000000000')] != '.':
                        print('WARNING: Instance ID not read correctly: ' + str(seq_id) + ' from ' + str(line))
            if seq_pos == -1:
                skipped += 1
                if check_missing_seq_ids:
                    pattern = re.compile(" [0123456789abcdef]{8}\-[0123456789abcdef]{4}-[0123456789abcdef]{4}-[0123456789abcdef]{4}-[0123456789abcdef]{12}")
                    if pattern.search(line):
                        print(line)
                continue
            line_parts = line.split(' ')
            line_content_start = line.find('-] ')
            if line_content_start != -1:
                line = line[(line_content_start + 2):]
            if line_content_start == -1:
                line_content_start = line.find('ERROR ')
                line = line[(line_content_start + 6):]
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
                if len(matches) > 0 and sorted(matches) == matches and (line[cur:] == '' or template_part == ''):
                    if template_id is not None:
                        print('WARNING: Templates ' + str(template_id) + ' and ' + str(i + 1) + ' both match line ' + line)
                    template_id = i + 1 # offset by 1 so that ID matches with template file
                    if line[cur:] != '':
                        params.append(line[cur:])
                    found_params = params # Store params found for matching template since params variable will be reset when checking next template
                    if early_stopping:
                        break
            if template_id is None:
                print('WARNING: No template matches ' + str(line) + '!')
            timestamp = -1
            if source != 'openstack_parisakalaki':
                # parisakalaki log data lack timestamps
                timestamp = datetime.datetime.strptime(line_parts[1] + ' ' + line_parts[2], '%Y-%m-%d %H:%M:%S.%f').replace(tzinfo=timezone.utc).timestamp()
            csv_line = str(cnt) + sep_csv + str(template_id) + sep_csv + str(seq_id) + sep_csv + str(timestamp) + sep_csv + str(label)
            if output_line:
                csv_line += sep_csv + str(line)
            if output_params:
                csv_line += sep_csv + sep_params.join(found_params)
            ext_file.write(csv_line + '\n')
    print('Skipped ' + str(skipped) + ' lines that did not contain any instance id.')
