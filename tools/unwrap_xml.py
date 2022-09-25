import sys
import re

if __name__ == '__main__':
    # helper script to unwrap xml corpus
    # use with `python unwarp_xml.py < input_file.xml > cleaned.txt`
    for line in sys.stdin:
        captured = re.findall(r'<seg.*>(.*?)<\/seg>', line)
        if len(captured):
            for x in captured:
                sys.stdout.write(x + '\n')
