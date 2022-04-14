import numpy as np
import sys

sent_path = sys.argv[1]
out_path = sys.argv[2]
max_len = 512

out_file = open(out_path, 'w+')

with open(sent_path) as sent_file:
    for line in sent_file:
        sent_1 = line[:line.index("|||")].split()
        sent_2 = line[line.index("|||"):].split()[1:]

        sent_1 = [word for word in sent_1 if word != '�' and word != '\u200b']
        sent_2 = [word for word in sent_2 if word != '�' and word != '\u200b']

        if len(sent_1) <= max_len and len(sent_2) <= max_len and len(sent_1) > 0 and len(sent_2) > 0:
            out_file.write(' '.join(sent_1) + ' ||| ' + ' '.join(sent_2) + '\n')

out_file.close()

