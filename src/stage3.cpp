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
    uint8_t *x0 = new uint8_t[2 * n * bytes_per_elem];
    memset(x0, 0, 2 * n * bytes_per_elem);
    for (size_t i = 0; i < n; ++i)
    {
      read_mnt_g1_montgomery(x0 + 2 * i * bytes_per_elem, inputs);
    }

    init_params_mnt4();
    uint8_t *res_x0 = reduce_g1(x0, n);

    uint8_t *res_x0_affine = g1_to_affine(res_x0);
    write_mnt_g1_montgomery(res_x0_affine, outputs);
    printG1(res_x0_affine);

    // mnt 4 G2
    uint8_t *y0 = new uint8_t[4 * n * bytes_per_elem];
    memset(y0, 0, 4 * n * bytes_per_elem);
    for (size_t i = 0; i < n; ++i)
    {
      read_mnt4_g2(y0 + 4 * i * bytes_per_elem, inputs);
    }

    // init_params_mnt4();
    // uint8_t *res_y0 = reduce_g2(y0, n);
    // write_mnt_g2(res_y0, outputs);

    // mnt 6 G1
    uint8_t *x1 = new uint8_t[2 * n * bytes_per_elem];
    memset(x1, 0, 2 * n * bytes_per_elem);
    for (size_t i = 0; i < n; ++i)
    {
      read_mnt_g1_montgomery(x1 + 2 * i * bytes_per_elem, inputs);
    }

    // init_params_mnt6();
    // uint8_t *res_x1 = reduce_g1(x1, n);
    // write_mnt_g1(res_x1, outputs);

    // mnt 6 G2
    uint8_t *y1 = new uint8_t[6 * n * bytes_per_elem];
    memset(y1, 0, 6 * n * bytes_per_elem);
    for (size_t i = 0; i < n; ++i)
    {
      read_mnt6_g2(y1 + 6 * i * bytes_per_elem, inputs);
    }

    // init_params_mnt6();
    // uint8_t *res_y1 = reduce_g2(y1, n);
    // write_mnt_g2(res_y1, outputs);

    delete[] x0;
    delete[] x1;
    delete[] y0;
    delete[] y1;
    // delete[] res_x0;
    // delete[] res_x1;
    // delete[] res_y0;
    // delete[] res_y1;
  }
}
