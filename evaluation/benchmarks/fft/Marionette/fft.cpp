#include <cstdint>
#define NPOINTS 256
#define NSTAGES 8

void kernel(float g_data_real[NPOINTS], float g_data_imag[NPOINTS],
            float g_coef_real[NPOINTS / 2], float g_coef_imag[NPOINTS / 2],
            int64_t groupsPerStage) {
  // groupsPerStage = 1;
  int64_t buttersPerGroup = NPOINTS / 2;

  // for (int64_t i = 0; i < NSTAGES; i++) {
  int64_t stage_offset = ((int64_t)1 << 0) - 1;

  // for (int64_t j = 0; j < groupsPerStage; j++) {
  int64_t j = 0;
  float Wr = g_coef_real[stage_offset + j];
  float Wi = g_coef_imag[stage_offset + j];

  int64_t base_idx = 2 * j * buttersPerGroup;

  // for (int64_t k = 0; k < buttersPerGroup; k++) {
  int64_t k = 0;
  int64_t idx1 = base_idx + k;
  int64_t idx2 = idx1 + buttersPerGroup;

  float dr1 = g_data_real[idx1];
  float di1 = g_data_imag[idx1];
  float dr2 = g_data_real[idx2];
  float di2 = g_data_imag[idx2];

  float temp_real = Wr * dr2 - Wi * di2;
  float temp_imag = Wi * dr2 + Wr * di2;

  g_data_real[idx2] = dr1 - temp_real;
  g_data_imag[idx2] = di1 - temp_imag;
  g_data_real[idx1] = dr1 + temp_real;
  g_data_imag[idx1] = di1 + temp_imag;
  // }
  // }

  //   groupsPerStage *= 2;
  //   buttersPerGroup /= 2;
  // }
}