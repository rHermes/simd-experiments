# -*- coding: utf-8 -*-
"""
So this is meant to be a simple
"""

import drawsvg as dw
from colour import Color

WIDTH = 1000
HEIGHT = 700

INS_TEXT_SIZE = 10

BLOCK_TEXT_SIZE = 8
BLOCK_WIDTH = 20
WHOLE_WIDTH = BLOCK_WIDTH*32


VERT_SPACE = 80

d = dw.Drawing(WIDTH, HEIGHT)

# White background.
d.append(dw.Rectangle(0, 0, WIDTH, HEIGHT, fill="white", stroke="red"))

# Non-rotated arrow
arrow = dw.Group(id='arrow')
arrow.append(dw.Line(0, 0, 40, 0, stroke_width=1))
arrow.append(dw.Lines(40, 0, 35, -5, 35, 5))

#d.append(dw.Use(arrow, 0, 0, stroke='black', fill='black'))


def createAVX2(names: list[tuple[str,int]], offset=None):
    if offset is not None:
        group = dw.Group(transform="translate({},{})".format(offset[0], offset[1]))
    else:
        group = dw.Group()
    
    for i, (name, subs, fill) in enumerate(names):
        block = dw.Group(transform="translate({})".format(i*BLOCK_WIDTH))
        
        block.append(dw.Rectangle(0, 0, BLOCK_WIDTH, BLOCK_WIDTH, fill=fill, stroke="black"))
        
        txt = dw.Text(name, BLOCK_TEXT_SIZE, BLOCK_WIDTH/2, BLOCK_WIDTH/2, center=True)
        if subs is not None:
            txt.append(dw.TSpan("{}".format(subs), style='font-size: 65%; baseline-shift: sub'))
    
        block.append(txt)
        
        
        group.append(block)
        
        
        
    return group

greenRange = list(Color("lightcyan").range_to(Color("steelblue"), 16))
pinkRange = list(Color("mistyrose").range_to(Color("tomato"), 16))
AA = [('A', i, c) for (i, c) in zip(range(0,32), greenRange + pinkRange)]
BB = [('B', i, "pink") for i in range(0,32)]


# Initial values

d.append(createAVX2(AA, (10, 20 + 0*VERT_SPACE)))

ag1 = dw.Group(transform="translate({}, {})".format(10 + WHOLE_WIDTH/2, 50 + 0*VERT_SPACE))
ag1.append(dw.Use(arrow, 0, 0, transform="rotate(90)", fill="black", stroke="black"))
ag1.append(dw.Text("_mm256_shuffle_epi8", INS_TEXT_SIZE, 5, 20, dominant_baseline="middle", text_anchor="left", font_family="monospace"))

d.append(ag1)



# Reshuffle so that they match the actual memory order
AA = AA[:16][::-1] + AA[16:][::-1]
d.append(createAVX2(AA, (10, 20 + 1*VERT_SPACE)))

ag2 = dw.Group(transform="translate({}, {})".format(10 + WHOLE_WIDTH/2, 50 + 1*VERT_SPACE))
ag2.append(dw.Use(arrow, 0, 0, transform="rotate(90)", fill="black", stroke="black"))
ag2.append(dw.Text("_mm256_permute2x128_si256(xs, xs, 0x01)", INS_TEXT_SIZE, 5, 20, dominant_baseline="middle", text_anchor="left", font_family="monospace"))

d.append(ag2)

# Flip lanes
AA = AA[16:] + AA[:16]
d.append(createAVX2(AA, (10, 20 + 2*VERT_SPACE)))


# OK, let's create a function for creating just the boxes.
d.save_svg("wow.svg")