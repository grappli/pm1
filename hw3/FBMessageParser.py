__author__ = 'Steffen'

import re

message_file = 'C:\Users\Steffen\Downloads\\facebook-grappli\html\messages.htm'
name = 'Egl'
outfile = name + '.txt'

with open(message_file) as f:
    text = f.read()
    m = re.findall(re.escape('class="user">' + name) + '.+?' + re.escape('<p>') + '(.+?)' + re.escape('</p>'),
                   text)
    with open(outfile, 'w') as w:
        w.write('\n'.join(m))