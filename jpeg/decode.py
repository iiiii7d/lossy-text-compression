import sys

from PIL import Image
from tqdm import tqdm

CHARS = """ !"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~\n"""

img = Image.open(sys.stdin.buffer)

for y in tqdm(range(img.height)):
    for x in range(img.width):
        pixel: tuple[int, int, int] = img.getpixel((x, y))
        """for val in pixel:
            index = val / 255 * (len(CHARS) - 1)
            encoded += CHARS[round(index)]"""
        val = (pixel[0]+pixel[1]+pixel[2])/3
        index = val / 255 * (len(CHARS) - 1)
        print(CHARS[round(index)], end="")