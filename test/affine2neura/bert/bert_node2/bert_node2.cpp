void bert_node2(
    const int input_indices[1][128],
    const float embedding_table[30522][768],
    float output[1][128][768]) {
    const int c30522 = 30522;
    const int c0_i64 = 0;
    
    for (int arg3 = 0; arg3 < 1; arg3++) {
        for (int arg4 = 0; arg4 < 128; arg4++) {
            for (int arg5 = 0; arg5 < 768; arg5++) {
                int index_i64 = input_indices[arg3][arg4];
                int index = static_cast<int>(index_i64);
                // Bound checking instead of assertions
                if (index >= c30522) {
                    index = c30522 - 1;  // Clamp to maximum valid index
                }
                if (index < c0_i64) {
                    index = c0_i64;  // Clamp to minimum valid index
                }
                float extracted_value = embedding_table[index][arg5];
                output[arg3][arg4][arg5] = extracted_value;
            }
        }
    }
}