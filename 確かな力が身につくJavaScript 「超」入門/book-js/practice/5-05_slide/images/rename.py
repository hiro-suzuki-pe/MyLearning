import glob
import os

folder = 'E:\\Prog\\確かな力がイにつくJavaScript 「超」入門\\book-js\\practice\\5-05_slide\\image'

os.chdir(folder)
images = glob.glob('*.jpg')
count = 0
for im in images:
    w = im.split('.')
    count += 1
    new_name = 'image{:02d}.{}'.format(count,w[1])
    os.rename(im, new_name)

