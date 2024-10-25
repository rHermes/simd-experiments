# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import drawsvg as dw
from colour import Color
from dataclasses import dataclass
import typing


WIDTH = 1000
HEIGHT = 700

INS_TEXT_SIZE = 10

BLOCK_TEXT_SIZE = 9

BLOCK_WIDTH = 20 / 8
BYTE_BLOCK_WIDTH = BLOCK_WIDTH*8

BLOCK_HEIGHT = BLOCK_WIDTH * 8

WHOLE_WIDTH = BLOCK_WIDTH*256


VERT_SPACE = 80

d = dw.Drawing(WIDTH, HEIGHT)

# White background.
d.append(dw.Rectangle(0, 0, WIDTH, HEIGHT, fill="white", stroke="red"))

# Non-rotated arrow
arrow = dw.Group(id='arrow')
arrow.append(dw.Line(0, 0, 40, 0, stroke_width=1))
arrow.append(dw.Lines(40, 0, 35, -5, 35, 5))

#d.append(dw.Use(arrow, 0, 0, stroke='black', fill='black'))


@dataclass
class SimdCell:
    value: str
    subscript: typing.Optional[str]
    fill: Color | dw.LinearGradient
    width: int = 8
    
    
    


def createAVX2(cells: list[SimdCell], offset=None):
    group = dw.Group()
    if offset is not None:
        group.args["transform"] = "translate({}, {})".format(offset[0], offset[1])
    
    shiftX = 0
    for cell in cells:
        block = dw.Group(transform="translate({})".format(shiftX))
        elemW = cell.width * BLOCK_WIDTH
        
        if isinstance(cell.fill, Color):
            block.append(dw.Rectangle(0, 0, elemW, BLOCK_HEIGHT, fill=cell.fill.get_hex(), stroke="black"))
        elif isinstance(cell.fill, dw.LinearGradient):
            block.append(dw.Rectangle(0, 0, elemW, BLOCK_HEIGHT, fill=cell.fill, stroke="black"))

        
        txt = dw.Text(cell.value, BLOCK_TEXT_SIZE, elemW*0.50, BLOCK_HEIGHT/2, center=True, font_family="monospace")
        
        if cell.subscript is not None:
            txt.append(dw.TSpan(cell.subscript, style='font-size: 65%; baseline-shift: sub'))
        
        block.append(txt)
        
        group.append(block)
        
        shiftX += elemW

    return group

LEFT_PADDING = 40


# Returns the group and how much to add to basePadding
# It's for binary operators, meaning they take two options.
def binaryAvxOperator(x, y, title, a, b, res, aName, bName, resName):
    opG = dw.Group(transform="translate({}, {})".format(x, y))
    
    opG.append(dw.Text(title, INS_TEXT_SIZE, LEFT_PADDING + WHOLE_WIDTH*0.5, -5, dominant_baseline="bottom", text_anchor="middle", font_family="monospace"))

    opG.append(createAVX2(a, (LEFT_PADDING, 0)))
    opG.append(dw.Text(aName, INS_TEXT_SIZE, LEFT_PADDING-5, BLOCK_HEIGHT*0.5, text_anchor="end", dominant_baseline="middle", font_family="monospace"))

    opG.append(createAVX2(b, (LEFT_PADDING, BLOCK_HEIGHT*1.2)))
    opG.append(dw.Text(bName, INS_TEXT_SIZE, LEFT_PADDING-5,  BLOCK_HEIGHT*1.2 + BLOCK_HEIGHT*0.5, text_anchor="end", dominant_baseline="middle", font_family="monospace"))

    opG.append(dw.Text("↓", INS_TEXT_SIZE*2, LEFT_PADDING + WHOLE_WIDTH*0.5, BLOCK_HEIGHT*2.2, dominant_baseline="hanging", text_anchor="middle", font_family="monospace"))

    
    opG.append(dw.Text(resName, INS_TEXT_SIZE, LEFT_PADDING-5,  BLOCK_HEIGHT*3 + BLOCK_HEIGHT*0.5, text_anchor="end", dominant_baseline="middle", font_family="monospace"))
    opG.append(createAVX2(res, (LEFT_PADDING, BLOCK_HEIGHT*3)))

    return (opG, VERT_SPACE*1.5)
    


greenRange = list(Color("lightcyan").range_to(Color("steelblue"), 16))
pinkRange = list(Color("mistyrose").range_to(Color("tomato"), 16))
AA = [SimdCell('A', str(i), c, 8) for (i, c) in zip(range(0,32), greenRange + pinkRange)]
# B = [('B', i, "pink") for i in range(0,32)]


reverseShuffle = [SimdCell('{}'.format(i % 16), None, Color("khaki" if i < 16 else "sandybrown")) for i in range(32)]
reverseShuffle.reverse()

CC =  AA[:16][::-1] + AA[16:][::-1]

blueGrad = dw.LinearGradient("0%", "50%", "100%", "50%", gradientUnits="objectBoundingBox")
blueGrad.add_stop("0%", CC[15].fill.get_hex(), )
blueGrad.add_stop("100%", CC[0].fill.get_hex())

redGrad = dw.LinearGradient("0%", "50%", "100%", "50%", gradientUnits="objectBoundingBox")
redGrad.add_stop("0%", CC[31].fill.get_hex(), )
redGrad.add_stop("100%", CC[16].fill.get_hex())


GG = [SimdCell("C", "128:0", blueGrad, 128), SimdCell("C", "256:128", redGrad, 128)]

DD = CC[16:] + CC[:16]



basePadding = 20

shuffleOp, spaceAdded = binaryAvxOperator(0, basePadding, "c = _mm256_shuffle_epi8(a, b)", AA, reverseShuffle, CC, "a", "b", "c")

d.append(shuffleOp)
basePadding += spaceAdded

flipOp, spaceAdded = binaryAvxOperator(0, basePadding, "d = _mm256_permute2x128_si256(c, c, 0x01)", GG, GG, [GG[1], GG[0]], "c", "c", "d")

d.append(flipOp)
basePadding += spaceAdded



d.save_svg("wow.svg")