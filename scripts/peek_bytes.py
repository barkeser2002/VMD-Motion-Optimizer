# VMD Motion Optimizer by Barış Keser (barkeser2002)
# License: GNU General Public License v3.0 (GPL-3.0)
# See LICENSE for details.

import sys, binascii, struct
p=sys.argv[1]
start=int(sys.argv[2]) if len(sys.argv)>2 else 0
n=int(sys.argv[3]) if len(sys.argv)>3 else 256
with open(p,'rb') as f:
    f.seek(start)
    data=f.read(n)
print('offset',start,'len',len(data))
print('hex',binascii.hexlify(data))
if start==0 and len(data)>=54:
    print('first30',data[:30])
    print('name20',data[30:50])
    print('bone_count_le', struct.unpack('<I', data[50:54])[0])
