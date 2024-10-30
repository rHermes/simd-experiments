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
HEIGHT = 2400

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

def showAvx(x, y, res, resName):
    opG = dw.Group(transform="translate({}, {})".format(x, y))
    
    # opG.append(dw.Text(title, INS_TEXT_SIZE, LEFT_PADDING + WHOLE_WIDTH*0.5, -5, dominant_baseline="bottom", text_anchor="middle", font_family="monospace"))

    # opG.append(dw.Text("↓", INS_TEXT_SIZE*2, LEFT_PADDING + WHOLE_WIDTH*0.5, BLOCK_HEIGHT*0.1, dominant_baseline="hanging", text_anchor="middle", font_family="monospace"))

    opG.append(dw.Text(resName, INS_TEXT_SIZE, LEFT_PADDING-5,  BLOCK_HEIGHT*0.5, text_anchor="end", dominant_baseline="middle", font_family="monospace"))
    opG.append(createAVX2(res, (LEFT_PADDING, BLOCK_HEIGHT*0)))

    return (opG, VERT_SPACE*0.5)

    
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

def calculateRunningSum(vecIn: list[SimdCell], inName: str, outName: str, rev: bool):
    G = dw.Group()
    
    outVec = [SimdCell(x.value, x.subscript, x.fill, x.width) for x in vecIn]
    
    if rev:
        for i in range(1,16):
            j = 15-i
            outVec[j].value = str(int(outVec[j].value) + int(outVec[j+1].value))
    
    opG, spaceAdded = unaryAvxOperator(0, 0, "{} = rimd::calcRunningSum<8, {}>({})".format(outName, rev, inName), vecIn, outVec, inName, outName)
    G.append(opG)
    
    return opG, spaceAdded, outVec



def visualizeSSE_v2_step_1(s1: str, updateVec: list[SimdCell], lagVec: list[SimdCell]):
    G = dw.Group()
    
    redStart = Color("mistyrose")
    redEnd = Color("tomato")
    
    blueStart = Color("lightcyan")
    blueEnd = Color("steelblue")
    
    cs = list(Color("LemonChiffon").range_to(Color("DarkKhaki"), 16))
    
    blueRange = list(blueStart.range_to(blueEnd, 16))
    redRange = list(redStart.range_to(redEnd, 16))
    
    AA = [SimdCell("'{}'".format(s1[i]), None, c, 8) for (i, c) in zip(range(0,16), redRange)]
    
    reverseShuffle = [SimdCell('{}'.format(i % 16), None, Color("khaki")) for i in range(16)]
    reverseShuffle.reverse()
    
    #  addVector = [SimdCell("79", None, Color("khaki" if i < 16 else "sandybrown")) for i in range(32)]
    
    oneVector = [SimdCell("'1'", None, Color("khaki"), 8) for i in range(16)]
    zeroVector = [SimdCell("0", None, Color("khaki"), 8) for i in range(16)]

    
    zeroSpotsVec = [SimdCell("-1" if AA[i].value == "'0'" else "0", None, AA[i].fill, 8) for i in range(16)]
    
    
    positionVec = [SimdCell(str(i+1), None, cs[i], 8) for i in range(16)]
    
    indexesVec = [SimdCell("0" if (zeroSpotsVec[i].value != "-1") else positionVec[i].value, None, zeroSpotsVec[i].fill, 8) for i in range(16)]
    
    
    basePadding = 20
    
    
    # Show of the beggining
    showUpdateOp, spaceAdded = showAvx(0, basePadding, updateVec, "update")
    G.append(showUpdateOp)
    basePadding += spaceAdded
    
    showLagOp, spaceAdded = showAvx(0, basePadding, lagVec, "lag")
    G.append(showLagOp)
    basePadding += spaceAdded
    
    basePadding += 20
    
    
    
    loadS1Op, spaceAdded = noneAvxOperator(0, basePadding, "chunk = _mm_loadu_si128(reinterpret_cast<__m128i const*>(inputString.data() + i))", AA, "chunk")
    G.append(loadS1Op)
    basePadding += spaceAdded
    
    subOp1, spaceAdded = binaryAvxOperator(0, basePadding, "zeroSpots = _mm_sub_epi8(chunk, _mm_set1_epi8('1'))", AA, oneVector, zeroSpotsVec, "chunk", "_mm_set1_epi8('1')", "zeroSpots")
    G.append(subOp1)
    basePadding += spaceAdded
    
    indexesOp, spaceAdded = binaryAvxOperator(0, basePadding, "indexes = _mm_and_si128(zeroSpots, POSITIONS)", zeroSpotsVec, positionVec, indexesVec, "zeroSpots", "POSITIONS", "indexes")
    G.append(indexesOp)
    basePadding += spaceAdded
    
    
    # revPsaOp, spaceAdded, revPsaVec = visualizeRunningSum(zeroSpotsVec, "zeroSpots", "revPsa", True)
    # revPsaOp.args["transform"] = "translate(0, {})".format(basePadding)
    # G.append(revPsaOp)
    # basePadding += spaceAdded
    
    revPsaOp, spaceAdded, revPsaVec = calculateRunningSum(indexesVec, "indexes", "revPsa", True)
    revPsaOp.args["transform"] = "translate(0, {})".format(basePadding)
    G.append(revPsaOp)
    basePadding += spaceAdded
    
    revNegZeroCountOp, spaceAdded, revNegZeroCountVec = calculateRunningSum(zeroSpotsVec, "zeroSpots", "revNegZeroCountVec", True)
    revNegZeroCountOp.args["transform"] = "translate(0, {})".format(basePadding)
    G.append(revNegZeroCountOp)
    basePadding += spaceAdded
    
    
    revZeroCountVec = [SimdCell(str(-int(x.value)), x.subscript, x.fill, x.width) for x in revNegZeroCountVec]
    negateOp, spaceAdded = binaryAvxOperator(0, basePadding, "revZeroCount = _mm_sub_epi8(_mm_setzero_si128(), revNegZeroCountVec)", zeroVector, revNegZeroCountVec, revZeroCountVec, "_mm_setzero_si128()","revNegZeroCount", "revZeroCount")
    G.append(negateOp)
    basePadding += spaceAdded
    
    onlyFirstByteVec = [SimdCell("0" if i != 0 else "255", None, cs[i], 8) for i in range(16)]
    psaVec = revPsaVec[0:1] + [SimdCell("0", None, revPsaVec[i].fill, 8) for i in range(1,16)]
    zeroCountVec = revZeroCountVec[0:1] + [SimdCell("0", None, revZeroCountVec[i].fill, 8) for i in range(1,16)]
    
    firstBytePsaOp, spaceAdded = binaryAvxOperator(0, basePadding, "psa = _mm_and_si128(revPsa, ONLY_FIRST_BYTE)", revPsaVec, onlyFirstByteVec, psaVec, "revPsa", "ONLY_FIRST_BYTE", "psa")
    G.append(firstBytePsaOp)
    basePadding += spaceAdded
    
    firstByteZeroCountOp, spaceAdded = binaryAvxOperator(0, basePadding, "psa = _mm_and_si128(revZeroCount, ONLY_FIRST_BYTE)", revZeroCountVec, onlyFirstByteVec, zeroCountVec, "revZeroCount", "ONLY_FIRST_BYTE", "zeroCount")
    G.append(firstByteZeroCountOp)
    basePadding += spaceAdded
    
    firstBytePsa64Vec = [SimdCell(psaVec[0].value, None, AA[0].fill, 64), SimdCell("0", None, AA[15].fill, 64)]
    tempUpdate1Vec = [SimdCell(str(int(firstBytePsa64Vec[0].value) + int(updateVec[0].value)), None, updateVec[0].fill, 64), updateVec[1]]
    
    tempUpdateOp, spaceAdded = binaryAvxOperator(0, basePadding, "update = _mm_add_epi64(update, psa)", updateVec, firstBytePsa64Vec, tempUpdate1Vec, "update", "psa", "update")
    G.append(tempUpdateOp)
    basePadding += spaceAdded
    
    # zeroCount32
    zeroCount32Vec = [SimdCell(zeroCountVec[0].value, None, zeroCountVec[0].fill, 32)] + [SimdCell("0", None, zeroCountVec[i*4].fill, 32) for i in range(1,4)]
    
    # Now to calculate triangle vector.
    triangVec164 = [SimdCell(str(int(zeroCount32Vec[i].value)**2), None, zeroCount32Vec[i].fill, 64) for i in range(0,4,2)]
    
    firstTriangOp, spaceAdded = binaryAvxOperator(0, basePadding, "triang = _mm_mul_epi32(zeroCount, zeroCount)", zeroCount32Vec, zeroCount32Vec, triangVec164, "zeroCount", "zeroCount", "triang")
    G.append(firstTriangOp)
    basePadding += spaceAdded
    
    
    triangVec1_32 = [SimdCell(triangVec164[0].value, None, zeroCountVec[0].fill, 32)] + [SimdCell("0", None, zeroCountVec[i*4].fill, 32) for i in range(1,4)]
    
    triangVec2_32 = [SimdCell(str(int(triangVec1_32[i].value) - int(zeroCount32Vec[i].value)), None, triangVec1_32[i].fill, 32) for i in range(4)]
    
    
    secondTriangOp, spaceAdded = binaryAvxOperator(0, basePadding, "triang = _mm_sub_epi32(triang, zeroCount)", triangVec1_32, zeroCount32Vec, triangVec2_32, "triang", "zeroCount", "triang")
    G.append(secondTriangOp)
    basePadding += spaceAdded
    
    
    triangVec3_32 = [SimdCell(str(int(triangVec2_32[i].value) >> 1), None, triangVec2_32[i].fill, 32) for i in range(4)]
    # one32Vec = [SimdCell("'1'", None, cs[i*4], 32) for i in range(4)]
    
    thirdTriangOp, spaceAdded = unaryAvxOperator(0, basePadding, "triang = _mm_srli_epi32(triang, 1)", triangVec2_32, triangVec3_32, "triang", "triang")
    G.append(thirdTriangOp)
    basePadding += spaceAdded
    
    
    triangVec3_64 = [SimdCell(triangVec3_32[0].value, None, AA[0].fill, 64), SimdCell("0", None, AA[15].fill, 64)]
    tempUpdate2Vec = [SimdCell(str(int(tempUpdate1Vec[0].value) - int(triangVec3_64[0].value)), None, updateVec[0].fill, 64), updateVec[1]]
    
    tempUpdate2Op, spaceAdded = binaryAvxOperator(0, basePadding, "update = _mm_sub_epi64(update, triang)", tempUpdate1Vec, triangVec3_64, tempUpdate2Vec, "update", "triang", "update")
    G.append(tempUpdate2Op)
    basePadding += spaceAdded
    
    # Now lag calculations
    tmpLagVec = [SimdCell(str(int(zeroCount32Vec[i].value)*int(lagVec[i].value)), None, lagVec[i].fill, 64) for i in range(0,4,2)]
    tmpLagMulOp, spaceAdded = binaryAvxOperator(0, basePadding, "tmpLag = _mm_mul_epi32(lag, zeroCount)", lagVec, zeroCount32Vec, tmpLagVec, "lag", "zeroCount", "tmpLag")
    G.append(tmpLagMulOp)
    basePadding += spaceAdded
    
    newUpdateVec = [SimdCell(str(int(tempUpdate2Vec[i].value) + int(tmpLagVec[i].value)), None, updateVec[i].fill, 64) for i in range(2)]
    
    finalUpdateOp, spaceAdded =  binaryAvxOperator(0, basePadding, "update = _mm_add_epi64(update, tmpLag)", tempUpdate2Vec, tmpLagVec, newUpdateVec, "update", "tmpLag", "update")
    G.append(finalUpdateOp)
    basePadding += spaceAdded
    


    tmpLag32Vec = [SimdCell(tmpLagVec[i].value if i == 0 else "0", None, lagVec[i].fill, 32) for i in range(4)]
    tmpLag1_32 = [SimdCell(str(int(lagVec[i].value) - int(zeroCount32Vec[i].value)), None, tmpLag32Vec[i].fill, 32) for i in range(4)]
    
    tmpLag2Op, spaceAdded = binaryAvxOperator(0, basePadding, "lag = _mm_sub_epi32(lag, zeroCount)", lagVec, zeroCount32Vec, tmpLag1_32, "lag", "zeroCount", "lag")
    G.append(tmpLag2Op)
    basePadding += spaceAdded
    
    
    tmpVec16 = [SimdCell("16", None, cs[i], 32) for i in range(4)]
    newLagVec = [SimdCell(str(int(tmpLag1_32[i].value) + int(tmpVec16[i].value)), None, tmpLag1_32[i].fill, 32) for i in range(4)]

    
    tmpLag3Op, spaceAdded = binaryAvxOperator(0, basePadding, "lag = _mm_add_epi32(lag,  _mm_set1_epi32(16))", tmpLag1_32, tmpVec16, newLagVec, "lag", "_mm_set1_epi32(16)", "lag")
    G.append(tmpLag3Op)
    basePadding += spaceAdded
    
    return G, basePadding, newUpdateVec, newLagVec




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
    
    lagCs = list(blueStart2.range_to(blueEnd2, 4))
    lagVec = [SimdCell("0" if i != 0 else "-1", None, lagCs[i], 32) for i in range(4)]
    
    updateVec = [SimdCell("0", None, lagCs[i], 64) for i in range(2)]



   #  s1G, mm1 = visualizeSSE_v2_step_1(s1, "1", blueStart1, blueEnd1, redStart1, redEnd1)
    # s1G, mm1 = visualizeSSE_v2_step_1(s1, "1", blueStart1, blueEnd1, redStart1, redEnd1)
    
    basePadding = 0

    s1G, spaceAdded, newUpdateVec, newLagVec = visualizeSSE_v2_step_1(s1, updateVec, lagVec)
    leftGroup.append(s1G)
    basePadding += spaceAdded
    
    s2G, spaceAdded, newUpdateVec, newLagVec = visualizeSSE_v2_step_1(s1, newUpdateVec, newLagVec)
    rightGroup.append(s2G)
    basePadding += spaceAdded
    
    
    # s2G = visualizeSSE_v2_step_1(s2)

    # s2G = visualizeAVX2_v2_step_1(s2, "2", Color("PaleGreen"), Color("SeaGreen"), Color("lavender"), Color("orchid"))

    # rightGroup.append(s2G)

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
