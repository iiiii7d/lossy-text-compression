import sys

from PIL import Image
from tqdm import tqdm

WIDTH = 512
CHARS = """ !"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~\n"""

text = ''.join(sys.stdin.readlines())

encoded = [CHARS.index(a) / (len(CHARS)-1) * 255 for a in text if a in CHARS]
encoded_groups = [encoded[pos:pos + 3] for pos in range(0, len(encoded), 3)]

img = Image.new('RGB', (WIDTH, int(len(encoded)//WIDTH+1)))

x = y = 0
"""for val in encoded_groups:
    while len(val) < 3:
        val.append(0.0)
    img.putpixel((x, y), (round(val[0]), round(val[1]), round(val[2])))
    x += 1
    if x == WIDTH:
        y += 1
        x = 0"""
for val in tqdm(encoded):
    img.putpixel((x, y), (round(val),)*3)
    x += 1
    if x == WIDTH:
        y += 1
        x = 0

img.save(sys.stdout, format='jpeg', quality=int(sys.argv[1] if len(sys.argv) > 1 else 100))

