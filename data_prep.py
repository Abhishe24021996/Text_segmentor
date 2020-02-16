import sys
import re
from tqdm import tqdm

def prepare_tag_data(filename):
    eng_lines = []
    with open(filename,'r',encoding='utf-8') as f:
        for line in tqdm(f):
            line = line.strip().replace('<s>','').replace('</t>','').split('</s> <t>')
            line = line[0].strip()
            line = re.sub('\s+',' ',line)
            eng_lines.append(line)
    with open('data/data.txt','w',encoding='utf-8') as f:
        for line in tqdm(eng_lines):
            f.write('%s\n'%line)
    
if __name__ == '__main__':
	print(sys.argv[1])
	prepare_tag_data(filename=sys.argv[1])
	print('done')