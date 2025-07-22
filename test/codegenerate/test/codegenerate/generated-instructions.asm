PE(0,2):
{
    Entry [East, R] => Once {
        {
            RETURN, [Local, R]
            NOP
        }
    }
}

PE(1,1):
{
    Entry [] => Once {
        {
            CONSTANT, IMM[1.000000e+00] -> [South, R]
        }
    }
}

PE(1,2):
{
    Entry [East, R], [South, R] => Once {
        {
            FADD, [South, R], [East, R] -> [East, R]
        }
    }
}

PE(2,2):
{
    Entry [] => Once {
        {
            CONSTANT, IMM[2.000000e+00] -> [East, R]
        }
    }
}

