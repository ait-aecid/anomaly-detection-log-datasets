import datetime
from datetime import timezone

templates = {}
output_line = False
output_params = False
source = "cfdr"
# Special separators need to be used when line and/or params are written to file as they contain commas
sep_csv = ';'
sep_params = '§'
service_names = set()
machine_ids = set()

events_allow_spaces = []

print('Get labels ...')
anomalous_sequences = set()
with open('thunderbird_' + source + '/tbird2', encoding='latin-1') as log_file:
    cnt = 0
    for line in log_file:
        cnt += 1
        if cnt % 5000000 == 0:
            print(str(cnt) + ' lines processed')
        label = line[:line.find(' ')]
        # Do not split whole line by spaces to speed up processing
        tmp1 = line[(len(label) + 1):line.find(' ', len(label) + 1)]
        tmp2 = line[(len(label) + 1 + len(tmp1) + 1):line.find(' ', len(label) + 1 + len(tmp1) + 1)]
        seq_id = line[(len(label) + 1 + len(tmp1) + 1 + len(tmp2) + 1):line.find(' ', len(label) + 1 + len(tmp1) + 1 + len(tmp2) + 1)]
        if label != '-':
            anomalous_sequences.add(seq_id)

print(anomalous_sequences)

print('Read lines ...')
with open('thunderbird_' + source + '/tbird2', encoding='latin-1') as log_file, open('templates/Thunderbird_templates.csv') as templates_file, open('thunderbird_' + source + '/parsed.csv', 'w+') as ext_file:
    ext_file.write('id;event_type;seq_id;time;label;eventlabel\n')
    i = 1
    for line in templates_file:
        template = line.strip('\n').rstrip(' ').split('<*>') # template is string after first appearance of comma
        if tuple(template) not in templates:
            templates[tuple(template)] = i
            i += 1
        else:
            print('WARNING: Identical template appears twice: ' + str(template))
    cnt = 0
    templates_list = list(templates.keys())
    for line in log_file:
        cnt += 1
        if cnt % 10000 == 0:
            print(str(cnt) + ' lines processed')
        line = line.strip('\n ')
        template_id = None
        line_parts = line.split(' ')
        eventlabel = line_parts[0]
        seq_id = line_parts[3]
        if eventlabel == '-':
            eventlabel = 'Normal'
        if seq_id in anomalous_sequences:
            label = "Anomaly"
        else:
            label = "Normal"
        found_params = []
        max_match_chars = -1
        best_template_id = -1
        if len(line_parts) > 8 and line_parts[8].endswith(':'):
            line = ' '.join(line_parts[9:])
        else:
            line = ' '.join(line_parts[8:])
        if line == '' or line == ' ': # Line is empty
            best_template_id = len(templates) + 1
            found_params = ['']
        else:
            template_loop_cnt = -1
            best_template_list_id = -1
            for template in templates_list:
                template_loop_cnt += 1
                if template[0] != '' and template[0][0] != line[0]:
                    continue
                matches = []
                params = []
                cur = 0
                starts_with_wildcard = False
                match_chars = 0
                for template_part in template:
                    pos = line.find(template_part, cur)
                    if pos == -1 or (' ' in line[cur:pos]): # and i not in events_allow_spaces): # or (not starts_with_wildcard and cur == 0 and pos != 0) or (i == 0 and not line.split(' ')[-1].split(':')[0].isdigit()):
                        matches = [] # Reset matches so that it counts as no match at all
                        break
                    matches.append(pos)
                    if line[cur:pos] != '':
                        params.append(line[cur:pos])
                    match_chars += len(template_part)
                    cur = pos + len(template_part)
                if len(matches) > 0 and (line[cur:] == '' or template_part == ''): # and (' ' not in line[cur:]): # or i in events_allow_spaces): # and (line[cur:] == '' or template_part == ''): # and '.' not in line[cur:]: and sorted(matches) == matches
                    template_id = templates[template] # + 1 # offset by 1 so that ID matches with template file
                    if line[cur:] != '':
                        params.append(line[cur:])
                    if match_chars > max_match_chars:
                        max_match_chars = match_chars
                        found_params = params # Store params found for matching template since params variable will be reset when checking next template
                        best_template_id = template_id
                        best_template_list_id = template_loop_cnt
            if best_template_id == -1:
                print('WARNING: No template matches "' + str(line) + '"')
            best_template = templates_list[best_template_list_id]
            del templates_list[best_template_list_id]
            templates_list = [best_template] + templates_list
        timestamp = datetime.datetime.strptime(line_parts[2] + ' ' + line_parts[6], '%Y.%m.%d %H:%M:%S').replace(tzinfo=timezone.utc).timestamp()
        csv_line = str(cnt) + sep_csv + str(best_template_id) + sep_csv + str(seq_id) + sep_csv + str(timestamp) + sep_csv + str(label) + sep_csv + str(eventlabel)
        if output_line:
            csv_line += sep_csv + str(line)
        if output_params:
            csv_line += sep_csv + sep_params.join(found_params)
        ext_file.write(csv_line + '\n')
