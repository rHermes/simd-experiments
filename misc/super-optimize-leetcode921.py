import z3
import operator

BV = 8
BV_MAX_U = z3.BitVecVal((1 << BV)-1, BV)
BV_MIN_U = z3.BitVecVal(0, BV)

BV_MAX_S = z3.BitVecVal((1 << (BV-1))-1, BV)
BV_MIN_S = z3.BitVecVal(-(1 << (BV-1)), BV)

# Left brace and right brace
LB = z3.BitVecVal(40, BV)
RB = z3.BitVecVal(41, BV)

if BV == 8:
    # This is left focused
    NEEDS = [
            (LB, 1),
            (RB, -1)
    ]

    NEEDS = [
            (RB, 1),
            (LB, -1)
    ]

    EXTRA_NEEDS = []
elif BV == 16:
    # prefix sum function
    # NEEDS = [
    #         ((LB<<8) + LB, 2),
    #         ((LB<<8) + RB, 0),
    #         ((RB<<8) + LB, 0),
    #         ((RB<<8) + RB, -2),
    #         ]
    # EXTRA_NEEDS = []

    # min function
    NEEDS = [
            # ((LB<<8) + LB, 0),
            # ((LB<<8) + RB, 0),
            ((LB<<8) + RB, -1),
            ((RB<<8) + RB, -2),
            # ((RB<<8) + RB, -2),
            ]

    EXTRA_NEEDS = [
            lambda op: (op((LB<<8) + LB) >= 0),
            lambda op: (op((RB<<8) + LB) >= 0),
            # lambda op: (z3.BVAddNoOverflow(op((LB<<8) + LB), 16, True)),
            # lambda op: (z3.BVAddNoOverflow(op((RB<<8) + LB), 16, True)),
    ]

print([(z3.simplify(x), y) for (x,y) in NEEDS])

# This file tries to hyper optimize the conversion from a
# a string of "(" and ")" to 1 and -1.


s = z3.SolverFor("BV")
# s = z3.Solver()

def saturedAddUnsigned(a, b):
    good = z3.BVAddNoOverflow(a, b, False)
    return z3.If(good, a+b, z3.BitVecVal(255, BV))

def saturatedAddSigned(a, b):
    goodOver = z3.BVAddNoOverflow(a, b, True)
    goodUnder = z3.BVAddNoUnderflow(a, b)
    return z3.If(goodOver, z3.If(goodUnder, a+b, z3.BitVecVal(-128, BV)), z3.BitVecVal(127, BV))

def maxU(a, b):
    return z3.If(z3.ULT(a, b), b, a)

def minU(a, b):
    return z3.If(z3.ULT(a,b), a, b)

def maxS(a, b):
    return z3.If(a < b, b, a)

def minS(a, b):
    return z3.If(a < b, a, b)

def mullo_16(a, b):
    if BV != 16:
        return a
    
    extA = z3.SignExt(16, a)
    extB = z3.SignExt(16, b)

    c = extA*extB
    d = z3.Extract(15, 0, c)
    # print(z3.simplify(d))
    # print(d.sort())
    return d




BIN_OPS = {
        "id": lambda x, y: x,
        # "inv": lambda x, y: ~x,

        # "cmp": lambda x, y: z3.If(x == y, BV_MAX_U, BV_MIN_U),
        # "minU": minU,
        # "maxU": maxU,
        # "maxS": maxS,
        # "minS": minS,

        # "addSatU": saturedAddUnsigned,
        # "addSatS": saturatedAddSigned,
        "add": operator.add,
        "sub": operator.sub,
        "and": operator.and_,
        "or": operator.or_,
        "xor": operator.xor,

        "llshift": z3.LShR,
        "lshift": operator.lshift,
        "rshift": operator.rshift,

        # "mullo16": mullo_16,
        # THese are the same 
        # "mullo16": operator.mul,
}


a, b, c, d, e, f = z3.BitVecs("a b c d e f", BV)

def check(s, op):
    s.push()
    for inp, out in NEEDS:
        s.add(op(inp) == out)

    for extra in EXTRA_NEEDS:
        s.add(extra(op))


    return s.check() == z3.sat
    

if __name__ == "__main__":
    for name1, op1 in BIN_OPS.items():
        for name2, op2 in BIN_OPS.items():
            for name3, op3 in BIN_OPS.items():
                if name3 != "id":
                    continue
                # So we have 2 possible combinations now.
                opc1 = lambda w: op2(op1(w, a), b)
                opc2 = lambda w: op2(w, op1(w, a))

                # opc3 = lambda w: op3(op1(w, a), op2(w, b))
                # opc4 = lambda w: op3(op1(w, a), c)
                # opc5 = lambda w: op3(op2(w, a), c)
                # opc6 = lambda w: op3(op2(w, a), c)


                if check(s, opc1):
                    s.add(e == opc1((LB<<8) + LB))
                    s.add(f == opc1((RB<<8) + LB))
                    s.check()
                    m = s.model()
                    # print("We did it: {}(w, {}) then {}({})".format(name1, m[a], name2, m[b] ))
                    print("We did it: {}(w, {:04X}) then {}({:04X}), with: op(\"((\") = {}, op(\"()\") = {}".format(
                        name1, m[a].as_long(), name2, m[b].as_long(), 
                        m[e].as_signed_long(), m[f].as_signed_long()))

                s.pop()

                if check(s, opc2):
                    m = s.model()
                    print("We did it: Y = {}(w, {}) then {}(w, Y)".format(name1, m[a].as_long(), name2))
                s.pop()
