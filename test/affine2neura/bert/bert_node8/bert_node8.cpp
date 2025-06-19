void bert_node8(
    const float input[1][128][1],
    float output[1][128][1]) {
    const float divisor = 768.0f;
    for (int arg3 = 0; arg3 < 1; arg3++) {
        for (int arg4 = 0; arg4 < 128; arg4++) {
            for (int arg5 = 0; arg5 < 1; arg5++) {
                float value = input[0][arg4][0];
                float result = value / divisor;
                output[arg3][arg4][arg5] = result;
            }
        }
    }
}