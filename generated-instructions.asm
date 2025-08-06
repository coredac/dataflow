PE(1,0):
{
    Entry => Loop {
        {
            ADD, [North, R], [NorthEast, R] -> [East, R], [Local, R]
        }
        {
            GRANT_PREDICATE, [Local, R], [East, R] -> [Local, R]
        }
    }
}

PE(1,1):
{
    Entry => Loop {
        {
            CONSTANT, IMM[10] -> [Local, R]
        }
        {
            GRANT_ALWAYS, [Local, R] -> [SouthEast, R]
        }
        {
            PHI, [Local, R], [North, R] -> [South, R]
        }
    }
}

PE(1,2):
{
    Entry => Loop {
        {
            CONSTANT, IMM[0] -> [Local, R]
        }
        {
            GRANT_ONCE, [Local, R] -> [South, R]
        }
    }
}

PE(2,0):
{
    Entry => Loop {
        {
            ICMP, [West, R], [NorthWest, R] -> [Local, R], [West, R]
        }
        {
            NOT, [Local, R] -> [NorthEast, R]
        }
        {
            GRANT_PREDICATE, [NorthEast, R], [Local, R] -> [Local, R]
        }
    }
}

PE(2,1):
{
    Entry => Loop {
        {
            CONSTANT, IMM[1] -> [Local, R]
        }
        {
            GRANT_ALWAYS, [Local, R] -> [SouthWest, R]
        }
        {
            GRANT_ONCE, [North, R] -> [Local, R]
        }
        {
            PHI, [Local, R], [Local, R] -> [East, R]
        }
    }
}

PE(2,2):
{
    Entry => Loop {
        {
            CONSTANT, IMM[3.000000e+00] -> [Local, R]
        }
        {
            CONSTANT, IMM[0.000000e+00] -> [South, R]
        }
        {
            GRANT_ALWAYS, [Local, R] -> [SouthEast, R]
        }
    }
}

PE(3,1):
{
    Entry => Loop {
        {
            FADD, [West, R], [NorthWest, R] -> [Local, R], [SouthWest, R]
        }
        {
            GRANT_PREDICATE, [Local, R], [SouthWest, R] -> [North, R]
        }
    }
}

PE(3,2):
{
    Entry => Once {
        {
            RETURN, [Local, R]
            NOP
        }
    }
}

