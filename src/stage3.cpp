#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>

#include "constants.hpp"
#include "io.hpp"
#include "reduce.hpp"
#include "utils.hpp"
#include "params.hpp"

void stage_3(FILE *inputs, FILE *outputs)
{
  size_t n;

  while (true)
  {
    size_t elts_read = fread((void *)&n, sizeof(size_t), 1, inputs);
    if (elts_read == 0)
    {
      break;
    }

    std::cerr << n << std::endl;

    // mnt 4 G1
    uint8_t *x0 = new uint8_t[3 * n * bytes_per_elem];
    memset(x0, 0, 3 * n * bytes_per_elem);
    for (size_t i = 0; i < n; ++i)
    {
      read_mnt4_g1_montgomery(x0 + 3 * i * bytes_per_elem, inputs);
    }

    init_params_mnt4();
    uint8_t *res_x0 = reduce_g1(x0, n);

    uint8_t *res_x0_affine = mnt4_g1_to_affine(res_x0);
    write_mnt4_g1_montgomery(res_x0_affine, outputs);
    //printG1(res_x0_affine);

    // mnt 4 G2
    uint8_t *y0 = new uint8_t[6 * n * bytes_per_elem];
    memset(y0, 0, 6 * n * bytes_per_elem);
    for (size_t i = 0; i < n; ++i)
    {
      read_mnt4_g2_montgomery(y0 + 6 * i * bytes_per_elem, inputs);
    }

    init_params_mnt4();
    uint8_t *res_y0 = reduce_mnt4_g2(y0, n);

    uint8_t *res_y0_affine = mnt4_g2_to_affine(res_y0);
    write_mnt4_g2_montgomery(res_y0_affine, outputs);
    //print_mnt4_G2(res_y0_affine);

    // mnt 6 G1
    uint8_t *x1 = new uint8_t[3 * n * bytes_per_elem];
    memset(x1, 0, 3 * n * bytes_per_elem);
    for (size_t i = 0; i < n; ++i)
    {
      read_mnt6_g1_montgomery(x1 + 3 * i * bytes_per_elem, inputs);
    }

    init_params_mnt6();
    uint8_t *res_x1 = reduce_g1(x1, n);

    uint8_t *res_x1_affine = mnt6_g1_to_affine(res_x1);
    write_mnt6_g1_montgomery(res_x1_affine, outputs);
    //printG1(res_x1_affine);

    // mnt 6 G2
    uint8_t *y1 = new uint8_t[9 * n * bytes_per_elem];
    memset(y1, 0, 9 * n * bytes_per_elem);
    for (size_t i = 0; i < n; ++i)
    {
      read_mnt6_g2_montgomery(y1 + 9 * i * bytes_per_elem, inputs);
    }

    init_params_mnt6();
    uint8_t *res_y1 = reduce_mnt6_g2(y1, n);

    uint8_t *res_y1_affine = mnt6_g2_to_affine(res_y1);
    write_mnt6_g2_montgomery(res_y1_affine, outputs);
    //print_mnt6_G2(res_y1_affine);

    delete[] x0;
    delete[] x1;
    delete[] y0;
    delete[] y1;
    delete[] res_x0;
    delete[] res_x1;
    delete[] res_y0;
    delete[] res_y1;
  }
}
