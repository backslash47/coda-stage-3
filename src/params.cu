#include <libff/algebra/curves/mnt753/mnt4753/mnt4753_pp.hpp>
#include <libff/algebra/curves/mnt753/mnt6753/mnt6753_pp.hpp>

#include "params.hpp"
#include "utils.hpp"

#include "gpu_constants.cuh"
#include "gpu_params.cuh"

using namespace libff;

void init_params_mnt4()
{
  HostParams params;
  params.set_mnt_mod((fixnum *)mnt4_modulus);
  params.set_mnt_non_residue((fixnum *)mnt4_non_residue);
  params.set_mnt_coeff_a((fixnum *)mnt4_g1_coeff_a);

  uint8_t *mnt4_g2_coeff_a2_c0 = new uint8_t[bytes_per_elem];
  memset(mnt4_g2_coeff_a2_c0, 0, bytes_per_elem);
  memcpy(mnt4_g2_coeff_a2_c0, (void *)mnt4753_G2::coeff_a.c0.as_bigint().data, io_bytes_per_elem);

  uint8_t *mnt4_g2_coeff_a2_c1 = new uint8_t[bytes_per_elem];
  memset(mnt4_g2_coeff_a2_c1, 0, bytes_per_elem);
  memcpy(mnt4_g2_coeff_a2_c1, (void *)mnt4753_G2::coeff_a.c1.as_bigint().data, io_bytes_per_elem);

  params.set_mnt_coeff_a2((fixnum *)mnt4_g2_coeff_a2_c0, (fixnum *)mnt4_g2_coeff_a2_c1);
  init_params(params);
}

void init_params_mnt6()
{
  HostParams params;
  params.set_mnt_mod((fixnum *)mnt6_modulus);
  params.set_mnt_non_residue((fixnum *)mnt6_non_residue);
  params.set_mnt_coeff_a((fixnum *)mnt6_g1_coeff_a);
  // FIXME: params.set_mnt_coeff_a2();
  init_params(params);
}
