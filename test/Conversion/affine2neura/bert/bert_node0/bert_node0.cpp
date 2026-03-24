void bert_node0(
    const int input[1][128],
    bool output[1][128]) {
    for (int arg3 = 0; arg3 < 1; arg3++) {
        for (int arg4 = 0; arg4 < 128; arg4++) {
            int value = input[0][arg4];
            bool result = (value > 0);
            output[arg3][arg4] = result;
        }
    }
}