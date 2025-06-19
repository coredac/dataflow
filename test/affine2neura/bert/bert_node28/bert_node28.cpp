void bert_node28(const float input_A[1][128][768],
                   const float input_B[1][768][768],
                   float output[1][128][768]) {

  for (int arg3 = 0; arg3 < 1; arg3++) {
    for (int arg4 = 0; arg4 < 128; arg4++) {
      for (int arg5 = 0; arg5 < 768; arg5++) {
        for (int arg6 = 0; arg6 < 768; arg6++) {
          float val_A = input_A[arg3][arg4][arg6];
          float val_B = input_B[arg3][arg6][arg5];
          float val_C = output[arg3][arg4][arg5];
          float mul_result = val_A * val_B;
          float add_result = val_C + mul_result;
          output[arg3][arg4][arg5] = add_result;
        }
      }
    }
  }
}