void bert_node3(const float input1[1][128][768],
                const float input2[1][128][768], float output[1][128][768]) {

  for (int arg3 = 0; arg3 < 1; arg3++) {
    for (int arg4 = 0; arg4 < 128; arg4++) {
      for (int arg5 = 0; arg5 < 768; arg5++) {
        float val1 = input1[0][arg4][arg5];
        float val2 = input2[0][arg4][arg5];
        float sum = val1 + val2;
        output[arg3][arg4][arg5] = sum;
      }
    }
  }
}