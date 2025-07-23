PE(0,2):
{
    Entry [West, R] => Once {
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
            CONSTANT, IMM[1.000000e+00] -> [North, R]
        }
    }
}

PE(1,2):
{
    Entry [North, R], [West, R] => Once {
        {
            FADD, [North, R], [West, R] -> [West, R]
        }
    }
}

PE(2,2):
{
    Entry [] => Once {
        {
            CONSTANT, IMM[2.000000e+00] -> [West, R]
        }
    }
}

