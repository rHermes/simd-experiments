# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import drawsvg as dw
from colour import Color

WIDTH = 1000
HEIGHT = 700

INS_TEXT_SIZE = 10

BLOCK_TEXT_SIZE = 9
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
        
        txt = dw.Text(name, BLOCK_TEXT_SIZE, BLOCK_WIDTH*0.50, BLOCK_WIDTH/2, center=True, font_family="monospace")
        if subs is not None:
            txt.append(dw.TSpan("{}".format(subs), style='font-size: 65%; baseline-shift: sub'))
    
        block.append(txt)
        
        
        group.append(block)
        
        
        
    return group

LEFT_PADDING = 40


# Returns the group and how much to add to basePadding
# It's for binary operators, meaning they take two options.
def binaryAvxOperator(x, y, title, a, b, res, aName, bName, resName):
    opG = dw.Group(transform="translate({}, {})".format(x, y))
    
    opG.append(dw.Text(title, INS_TEXT_SIZE, LEFT_PADDING + WHOLE_WIDTH*0.5, -5, dominant_baseline="bottom", text_anchor="middle", font_family="monospace"))

    opG.append(createAVX2(a, (LEFT_PADDING, 0)))
    opG.append(dw.Text(aName, INS_TEXT_SIZE, LEFT_PADDING-5, BLOCK_WIDTH*0.5, text_anchor="end", dominant_baseline="middle", font_family="monospace"))

    opG.append(createAVX2(b, (LEFT_PADDING, BLOCK_WIDTH*1.2)))
    opG.append(dw.Text(bName, INS_TEXT_SIZE, LEFT_PADDING-5,  BLOCK_WIDTH*1.2 + BLOCK_WIDTH*0.5, text_anchor="end", dominant_baseline="middle", font_family="monospace"))

    opG.append(dw.Text("↓", INS_TEXT_SIZE*2, LEFT_PADDING + WHOLE_WIDTH*0.5, BLOCK_WIDTH*2.2, dominant_baseline="hanging", text_anchor="middle", font_family="monospace"))

    
    opG.append(dw.Text(resName, INS_TEXT_SIZE, LEFT_PADDING-5,  BLOCK_WIDTH*3 + BLOCK_WIDTH*0.5, text_anchor="end", dominant_baseline="middle", font_family="monospace"))
    opG.append(createAVX2(res, (LEFT_PADDING, BLOCK_WIDTH*3)))

    return (opG, VERT_SPACE*1.5)
    


greenRange = list(Color("lightcyan").range_to(Color("steelblue"), 16))
pinkRange = list(Color("mistyrose").range_to(Color("tomato"), 16))
AA = [('A', i, c) for (i, c) in zip(range(0,32), greenRange + pinkRange)]
BB = [('B', i, "pink") for i in range(0,32)]


reverseShuffle = [('{}'.format(i % 16), None, "khaki" if i < 16 else "sandybrown") for i in range(32)]
reverseShuffle.reverse()

CC =  AA[:16][::-1] + AA[16:][::-1]
DD = CC[16:] + CC[:16]



basePadding = 20

shuffleOp, spaceAdded = binaryAvxOperator(0, basePadding, "c = _mm256_shuffle_epi8(a, b)", AA, reverseShuffle, CC, "a", "b", "c")

d.append(shuffleOp)
basePadding += spaceAdded

flipOp, spaceAdded = binaryAvxOperator(0, basePadding, "d = _mm256_permute2x128_si256(c, c, 0x01)", CC, CC, DD, "c", "c", "d")

d.append(flipOp)
basePadding += spaceAdded



d.save_svg("wow.svg")