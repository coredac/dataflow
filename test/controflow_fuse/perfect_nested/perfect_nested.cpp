void bert_node1(
    bool input[1][1][1][1][1][128], 
    bool output[1][1][128][1][1][128]) {
    
    for (int arg3 = 0; arg3 < 1; arg3++) {
        for (int arg4 = 0; arg4 < 1; arg4++) {
            for (int arg5 = 0; arg5 < 128; arg5++) {
                for (int arg6 = 0; arg6 < 1; arg6++) {
                    for (int arg7 = 0; arg7 < 1; arg7++) {
                        for (int arg8 = 0; arg8 < 128; arg8++) {
                            bool value = input[arg3][arg4][0][arg6][arg7][arg8];
                            output[arg3][arg4][arg5][arg6][arg7][arg8] = value;
                        }
                    }
                }
            }
        }
    }
}