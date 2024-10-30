# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import drawsvg as dw
from colour import Color
from dataclasses import dataclass
import typing

def blend(a: Color, b: Color) -> Color:
    return list(a.range_to(b, 3))[1]

def unsignedToSigned(n, byte_count): 
  return int.from_bytes(n.to_bytes(byte_count, 'little', signed=False), 'little', signed=True)

def signedToUnsigned(n, byte_count): 
  return int.from_bytes(n.to_bytes(byte_count, 'little', signed=True), 'little', signed=False)

WIDTH = 1700
HEIGHT = 1400

INS_TEXT_SIZE = 10

BLOCK_TEXT_SIZE = 9

BLOCK_WIDTH = 20 / 8
BYTE_BLOCK_WIDTH = BLOCK_WIDTH*8

BLOCK_HEIGHT = BLOCK_WIDTH * 8

WHOLE_WIDTH = BLOCK_WIDTH*128


VERT_SPACE = 80


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

LEFT_PADDING = 130


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

def unaryAvxOperator(x, y, title, a, res, aName, resName):
    opG = dw.Group(transform="translate({}, {})".format(x, y))
    
    opG.append(dw.Text(title, INS_TEXT_SIZE, LEFT_PADDING + WHOLE_WIDTH*0.5, -5, dominant_baseline="bottom", text_anchor="middle", font_family="monospace"))

    opG.append(createAVX2(a, (LEFT_PADDING, 0)))
    opG.append(dw.Text(aName, INS_TEXT_SIZE, LEFT_PADDING-5, BLOCK_HEIGHT*0.5, text_anchor="end", dominant_baseline="middle", font_family="monospace"))

    opG.append(dw.Text("↓", INS_TEXT_SIZE*2, LEFT_PADDING + WHOLE_WIDTH*0.5, BLOCK_HEIGHT*1.2, dominant_baseline="hanging", text_anchor="middle", font_family="monospace"))

    opG.append(dw.Text(resName, INS_TEXT_SIZE, LEFT_PADDING-5,  BLOCK_HEIGHT*2 + BLOCK_HEIGHT*0.5, text_anchor="end", dominant_baseline="middle", font_family="monospace"))
    opG.append(createAVX2(res, (LEFT_PADDING, BLOCK_HEIGHT*2)))

    return (opG, VERT_SPACE*1.2)


def noneAvxOperator(x, y, title, res, resName):
    opG = dw.Group(transform="translate({}, {})".format(x, y))
    
    opG.append(dw.Text(title, INS_TEXT_SIZE, LEFT_PADDING + WHOLE_WIDTH*0.5, -5, dominant_baseline="bottom", text_anchor="middle", font_family="monospace"))

    opG.append(dw.Text("↓", INS_TEXT_SIZE*2, LEFT_PADDING + WHOLE_WIDTH*0.5, BLOCK_HEIGHT*0.1, dominant_baseline="hanging", text_anchor="middle", font_family="monospace"))

    opG.append(dw.Text(resName, INS_TEXT_SIZE, LEFT_PADDING-5,  BLOCK_HEIGHT*1 + BLOCK_HEIGHT*0.5, text_anchor="end", dominant_baseline="middle", font_family="monospace"))
    opG.append(createAVX2(res, (LEFT_PADDING, BLOCK_HEIGHT*1)))

    return (opG, VERT_SPACE*0.9)
    
    
def visualizeRunningSum(vecIn: list[SimdCell], inName: str, outName: str, rev: bool):
    G = dw.Group()
    basePadding = 20
    
    def shiftIt(inp, inpName, oName, times):
        nonlocal basePadding
        
        if rev:
            g = inp[times:] + [SimdCell("0", None, Color("khaki"), 8)]*times
        else:
            g = [SimdCell("0", None, Color("khaki"), 8)]*times +  inp[:(16-times)]
        
        if rev:
            shiftTitle = "{} = _mm_bsrli_si128({}, {})".format(oName, inpName, times)
        else:
            shiftTitle = "{} = _mm_bslli_si128({}, {})".format(oName, inpName, times)

        
        oper, spaceAdded = unaryAvxOperator(0, basePadding, shiftTitle, inp, g, inpName, oName)
        G.append(oper)
        basePadding += spaceAdded
        return g
    
    def addIt(inpA, inpAName, inpB, inpBName, oName):
        nonlocal basePadding
        
        res = [SimdCell(str(int(inpA[i].value) + int(inpB[i].value)), None, inpA[i].fill, 8) for i in range(16)]
        
        oper, spaceAdded = binaryAvxOperator(0, basePadding, "{} = _mm_add_epi8({}, {})".format(oName, inpAName, inpBName), inpA, inpB,  res, inpAName, inpBName, oName)
        G.append(oper)
        basePadding += spaceAdded
        return res
        
        
        
    
    shift1Vec = shiftIt(vecIn, inName, "shift1", 1)
    temp1Vec = addIt(vecIn, inName, shift1Vec, "shift1", "temp1")
    
    shift2Vec = shiftIt(temp1Vec, "temp1", "shift2", 2)
    temp2Vec = addIt(temp1Vec, "temp1", shift2Vec, "shift2", "temp2")
    
    shift4Vec = shiftIt(temp2Vec, "temp2", "shift4", 4)
    temp3Vec = addIt(temp2Vec, "temp2", shift4Vec, "shift4", "temp3")
    
    shift8Vec = shiftIt(temp3Vec, "temp3", "shift8", 8)
    vecOut = addIt(temp3Vec, "temp3", shift8Vec, "shift8", outName)

    return G, basePadding, vecOut
    



def visualizeSSE_v2_step_1(s1: str):
    G = dw.Group()
    
    redStart = Color("mistyrose")
    redEnd = Color("tomato")
    
    blueStart = Color("lightcyan")
    blueEnd = Color("steelblue")
    
    blueRange = list(blueStart.range_to(blueEnd, 16))
    redRange = list(redStart.range_to(redEnd, 16))
    
    AA = [SimdCell("'{}'".format(s1[i]), None, c, 8) for (i, c) in zip(range(0,16), redRange)]
    
    reverseShuffle = [SimdCell('{}'.format(i % 16), None, Color("khaki")) for i in range(16)]
    reverseShuffle.reverse()
    
    addVector = [SimdCell("79", None, Color("khaki" if i < 16 else "sandybrown")) for i in range(32)]
    
    oneVector = [SimdCell("'1'", None, Color("khaki"), 8) for i in range(16)]
    
    zeroSpotsVec = [SimdCell("-1" if AA[i].value == "'0'" else "0", None, AA[i].fill, 8) for i in range(16)]
    
    CC =  AA[:16][::-1] + AA[16:][::-1]
    
    blueGrad = dw.LinearGradient("0%", "50%", "100%", "50%", gradientUnits="objectBoundingBox")
    blueGrad.add_stop("0%", CC[15].fill.get_hex())
    blueGrad.add_stop("100%", CC[0].fill.get_hex())
    
    # redGrad = dw.LinearGradient("0%", "50%", "100%", "50%", gradientUnits="objectBoundingBox")
    # redGrad.add_stop("0%", CC[31].fill.get_hex())
    # redGrad.add_stop("100%", CC[16].fill.get_hex())
    
    goodThing1 = 0
    for x in reversed(CC[:16]):
        goodThing1 <<= 8
        goodThing1 += ord(x.value[1])
    
    goodThing2 = 0
    for x in reversed(CC[16:]):
        goodThing2 <<= 8
        goodThing2 |= ord(x.value[1])
    
    
    # GG = [SimdCell("0x{:016X}".format(goodThing1), None, blueGrad, 128), SimdCell("0x{:016X}".format(goodThing2), None, redGrad, 128)]

    # DD = CC[16:] + CC[:16]
    # value1 = [SimdCell("{}".format(79 + ord(x.value[1])), None, x.fill, 8) for x in DD]
    
    basePadding = 20
    
    loadS1Op, spaceAdded = noneAvxOperator(0, basePadding, "chunk = _mm_loadu_si128(reinterpret_cast<__m128i const*>(inputString.data() + i))", AA, "chunk")
    basePadding += spaceAdded    
    G.append(loadS1Op)
    
    subOp1, spaceAdded = binaryAvxOperator(0, basePadding, "zeroSpots = _mm_sub_epi8(chunk, _mm_set1_epi8('1'))", AA, oneVector, zeroSpotsVec, "chunk", "_mm_set1_epi8('1')", "zeroSpots")
    G.append(subOp1)
    basePadding += spaceAdded
    
    revPsaOp, spaceAdded, revPsaVec = visualizeRunningSum(zeroSpotsVec, "zeroSpots", "revPsa", True)
    revPsaOp.args["transform"] = "translate(0, {})".format(basePadding)
    G.append(revPsaOp)
    basePadding += spaceAdded
    
    
    
    # shuffleOp, spaceAdded = binaryAvxOperator(0, basePadding, "revChunk{0} = _mm256_shuffle_epi8(chunk{0}, REVERSE_SHUFFLE_MASK)".format(suf), AA, reverseShuffle, CC, "chunk" + suf,  "REVERSE_SHUFFLE_MASK", "revChunk" + suf)
    # G.append(shuffleOp)
    # basePadding += spaceAdded
    
    # flipOp, spaceAdded = binaryAvxOperator(0, basePadding, "revChunk{0} = _mm256_permute2x128_si256(revChunk{0}, revChunk{0}, 0x01)".format(suf), GG, GG, [GG[1], GG[0]], "revChunk" + suf,  "revChunk" + suf, "realRevChunk" + suf)
    # G.append(flipOp)
    # basePadding += spaceAdded
    
    # addOp, spaceAdded = binaryAvxOperator(0, basePadding, "values{0} = _mm256_add_epi8(realRevChunk{0}, ADD_MASK)".format(suf), DD, addVector, value1, "realRevChunk" + suf, "ADD_MASK", "values" + suf)
    # G.append(addOp)
    # basePadding += spaceAdded


    # # Move mask
    # MM1 = [SimdCell("{}".format(int(x.value)>>7), None, x.fill, 3) for x in value1]
    
    # moveMaskOp, spaceAdded = unaryAvxOperator(0, basePadding, "mm{0} = _mm256_movemask_epi8(values{0})".format(suf), value1, MM1, "values" + suf, "mm" + suf)
    # G.append(moveMaskOp)
    # basePadding += spaceAdded
    
    
    return G


def visualizeAVX2_v2_step_2(mm1, mm2):
    G = dw.Group()
    
    #redStart1 = Color("mistyrose")
    # redEnd1 = Color("tomato")
    blueStart1 = Color("lightcyan")
    blueEnd1 = Color("steelblue")
    
    mm1Value = int("".join([x.value for x in reversed(mm1)]), base=2)
    mm2Value = int("".join([x.value for x in reversed(mm2)]), base=2)
    mm3Value = mm1Value + mm2Value
    mmCarryOut = mm3Value >> 32
    mm3Value = mm3Value & 0xFFFFFFFF

    mm3Str = "{:032b}".format(mm3Value)
    mm3Str = mm3Str[::-1]

    mm3 = [SimdCell(mm3Str[i], None, blend(mm1[i].fill, mm2[i].fill), 3) for i in range(32)]

    basePadding = 20
    addCarryOp, spaceAdded = binaryAvxOperator(0, basePadding, "carry = _addcarry_u32(carry, mm1, mm2, &mm3)", mm1, mm2, mm3, "mm1", "mm2", "mm3")
    basePadding += spaceAdded
    G.append(addCarryOp)

    return G, mm3

def visualizeAVX2_v2_step_3(mm3):
    G = dw.Group()

    mm3Value = int("".join([x.value for x in mm3[::-1]]), base=2)
    mm3_32bit_val = "0x{:08X}".format(mm3Value)
        
    cs = list(Color("LemonChiffon").range_to(Color("DarkKhaki"), 4))
    extractShuffle = [SimdCell('{}'.format(i // 64), None, cs[i//64], 8) for i in range(0, 256, 8)]
    extractShuffle.reverse()

    cs2 = list(Color("LemonChiffon").range_to(Color("DarkKhaki"), 8))
    completeMask = [SimdCell("{:02X}".format(0xFF ^ (1<<i)), None, cs2[i], 8) for i in range(8)]*4
    completeMask.reverse()

    redGrad = dw.LinearGradient("0%", "50%", "100%", "50%", gradientUnits="objectBoundingBox")
    redGrad.add_stop("0%", mm3[31].fill.get_hex())
    # redGrad.add_stop("49%", mm3[16].fill.get_hex())
    redGrad.add_stop("50%", mm3[16].fill.get_hex())
    redGrad.add_stop("50%", mm3[15].fill.get_hex())
    redGrad.add_stop("100%", mm3[0].fill.get_hex())

    res1 = [SimdCell(mm3_32bit_val, None, redGrad, 32) for _ in range(8)]

    basePadding = 20
    setEpi32Op, spaceAdded = unaryAvxOperator(0, basePadding, "res1 = _mm256_set1_epi32(mm3)", mm3, res1, "mm3", "res1")
    basePadding += spaceAdded
    G.append(setEpi32Op)
        
    # ok we are need to take the upper bytes on each
    res2 = []
    for i in range(32):
        j = i // 8
        startText = 2 + (3-j)*2
        endText = startText + 2
        val = mm3_32bit_val[startText:endText]
        upperBit = j*8 + 7
        lowerBit = j*8

        cellGrad = dw.LinearGradient("0%", "50%", "100%", "50%", gradientUnits="objectBoundingBox")
        cellGrad.add_stop("0%", mm3[upperBit].fill.get_hex())
        cellGrad.add_stop("100%", mm3[lowerBit].fill.get_hex())

        res2.append(SimdCell(val, None, cellGrad, 8))

    res2.reverse()



    resShuffleOp, spaceAdded = binaryAvxOperator(0, basePadding, "res2 = _mm256_shuffle_epi8(res1, byteExtractMask)", res1, extractShuffle, res2, "res1", "byteExtractMask", "res2")
    basePadding += spaceAdded
    G.append(resShuffleOp)
    
    res3 = [SimdCell("{:02X}".format(int(res2[i].value, base=16) | int(completeMask[i].value, base=16)), None, mm3[31-i].fill, 8) for i in range(32)]
    
    resCompleteOp, spaceAdded = binaryAvxOperator(0, basePadding, "res3 = _mm256_or_si256(res2, bitCompleteMask)", res2, completeMask, res3, "res2", "bitCompleteMask", "res3")
    basePadding += spaceAdded
    G.append(resCompleteOp)

    res4 = [SimdCell("FF" if completeMask[i].value == res3[i].value else "0", None, res3[i].fill, 8) for i in range(32)]
    
    reCmpOp, spaceAdded = binaryAvxOperator(0, basePadding, "res4 = _mm256_cmpeq_epi8(res3, bitCompleteMask)", res3, completeMask, res4, "res3", "bitCompleteMask", "res4")
    basePadding += spaceAdded
    G.append(reCmpOp)

    charsOut = [SimdCell("'{}'".format(chr((int(res4[i].value, base=16) + ord('1')) % 256)), None, res4[i].fill, 8) for i in range(32)]
    AsciiOne = [SimdCell("'1'", None, Color("khaki" if i < 16 else "sandybrown"), 8) for i in range(32)]
    
    reAddOp, spaceAdded = binaryAvxOperator(0, basePadding, "charsOut = _mm256_add_epi8(res4, ASCII_ONE)", res4, AsciiOne, charsOut, "res4", "ASCII_ONE", "charsOut")
    basePadding += spaceAdded
    G.append(reAddOp)


    return G


def showBitCompleteMask():
    group = dw.Group()
    
    
    cs2 = list(Color("LemonChiffon").range_to(Color("DarkKhaki"), 8))
    completeMask = [SimdCell("{:02X}".format(0xFF ^ (1<<i)), None, cs2[i], 8) for i in range(8)]
    completeMask.reverse()
    
    group.append(dw.Text("bitCompleteMask", BLOCK_TEXT_SIZE*2, 0, 0, font_family="monospace")) 
    
    shiftY = 10
    for cell in completeMask:
        block = dw.Group(transform="translate(0, {})".format(shiftY))
        elemW = cell.width * BLOCK_WIDTH
        
        block.append(dw.Rectangle(0, 0, elemW, BLOCK_HEIGHT, fill=cell.fill.get_hex(), stroke="black"))
        txt = dw.Text(cell.value, BLOCK_TEXT_SIZE, elemW*0.50, BLOCK_HEIGHT/2, center=True, font_family="monospace")        
        block.append(txt)
        
        writeText = dw.Text("{:08b}".format(int(cell.value, base=16)), BLOCK_TEXT_SIZE, elemW + 10 , BLOCK_HEIGHT/2, dominant_baseline="Middle", text_anchor="start", font_family="monospace")
        block.append(writeText)
        
        group.append(block)
        
        shiftY += elemW

    return group


    

def visualizeSSE_v2(s):
    s1 = s[:16]
    s2 = s[16:32]
    
    G = dw.Group()

    leftGroup = dw.Group()
    rightGroup = dw.Group(transform="translate(800)")

    G.append(leftGroup)
    G.append(rightGroup)



    redStart2 = Color("palegreen")
    redEnd2 = Color("seagreen")
    blueStart2 = Color("lavender")
    blueEnd2 = Color("orchid")

   #  s1G, mm1 = visualizeSSE_v2_step_1(s1, "1", blueStart1, blueEnd1, redStart1, redEnd1)
    # s1G, mm1 = visualizeSSE_v2_step_1(s1, "1", blueStart1, blueEnd1, redStart1, redEnd1)

    s1G = visualizeSSE_v2_step_1(s1)
    s2G = visualizeSSE_v2_step_1(s2)

    # s2G = visualizeAVX2_v2_step_1(s2, "2", Color("PaleGreen"), Color("SeaGreen"), Color("lavender"), Color("orchid"))

    leftGroup.append(s1G)
    rightGroup.append(s2G)

    # stepTwoG, mm3 = visualizeAVX2_v2_step_2(mm1, mm2)
    # stepTwoG.args["transform"] = "translate(0, 600)"
    # leftGroup.append(stepTwoG)

    # stepThreeG = visualizeAVX2_v2_step_3(mm3)
    # stepThreeG.args["transform"] = "translate(0, 800)"
    # leftGroup.append(stepThreeG)
    
    # bitCmpMask = showBitCompleteMask()
    # bitCmpMask.args["transform"] = "translate(30, 950) scale(1.4)"
    # rightGroup.append(bitCmpMask)
    
    return G



d = dw.Drawing(WIDTH, HEIGHT)
# .append(dw.Rectangle(0, 0, WIDTH, HEIGHT, fill="white", stroke="red"))

# d.append(visualizeAVX2_v2('10100101010110011001001011000000', '11001000101100000000110101010100'))
d.append(visualizeSSE_v2('10100101010110011001001011000000'))
# print(d.as_svg())

# White background.
# d.save_svg("wow.svg")
d.save_html("leetcode-2938-sse-v2.html")
