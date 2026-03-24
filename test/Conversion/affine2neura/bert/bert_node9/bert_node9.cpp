void bert_node9(
    const float input[1][128][768], 
    double output[1][128][768]) {
    for (int arg3 = 0; arg3 < 1; arg3++) {
        for (int arg4 = 0; arg4 < 128; arg4++) {
            for (int arg5 = 0; arg5 < 768; arg5++) {
                float value = input[0][arg4][arg5];
                double extended_value = static_cast<double>(value);
                output[arg3][arg4][arg5] = extended_value;
            }
        }
    }
}