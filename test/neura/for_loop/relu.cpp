// RUN: mlir-neura-opt %s | FileCheck %s

#include <stdio.h>

#define N 32

float input[N] = {
  -1.0, 0.0, 1.0, 2.0, -3.0, 4.0, -5.0, 6.0,
  -7.0, 8.0, -9.0, 10.0, -11.0, 12.0, -13.0, 14.0,
  -15.0, 16.0, -17.0, 18.0, -19.0, 20.0, -21.0, 22.0,
  -23.0, 24.0, -25.0, 26.0, -27.0, 28.0, -29.0, 30.0
};

float output[N];

void kernel(float input[], float output[]);

int main(){
  //init output
  for(int i=0; i<N; i++){
    output[i]=0.0;
  }
 
  kernel(intput, output);

  //print outputs
  for(int i=0; i<N; i++){
    print("output[%d] = %f\n", i, output[i]); 
  }

  return 0;
}

void kernel(float input[], float output []){
  for(int i = 0; i<N; ++i){
    if(input[i]>0){
      output[i] += input[i];
    } else {
      output[i] += 0;
    }
  }
}
