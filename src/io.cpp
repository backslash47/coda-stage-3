#include <libff/algebra/curves/mnt753/mnt4753/mnt4753_pp.hpp>
#include <libff/algebra/curves/mnt753/mnt6753/mnt6753_pp.hpp>

#include "io.hpp"
#include "constants.hpp"

using namespace libff;

void read_mnt_fq(uint8_t *dest, FILE *inputs)
{
  fread((void *)(dest), io_bytes_per_elem * sizeof(uint8_t), 1, inputs);
}

void read_mnt_fq_montgomery(uint8_t *dest, FILE *inputs)
{
  Fq<mnt4753_pp> x;
  fread((void *)(x.mont_repr.data), io_bytes_per_elem * sizeof(uint8_t), 1, inputs);
  memcpy(dest, (uint8_t *)x.as_bigint().data, io_bytes_per_elem);
}

void read_mnt_fq2(uint8_t *dest, FILE *inputs)
{
  read_mnt_fq(dest, inputs);
  read_mnt_fq(dest + bytes_per_elem, inputs);
}

void read_mnt_fq3(uint8_t *dest, FILE *inputs)
{
  read_mnt_fq(dest, inputs);
  read_mnt_fq(dest + bytes_per_elem, inputs);
  read_mnt_fq(dest + 2 * bytes_per_elem, inputs);
}

void read_mnt_g1_montgomery(uint8_t *dest, FILE *inputs)
{
  read_mnt_fq_montgomery(dest, inputs);
  read_mnt_fq_montgomery(dest + bytes_per_elem, inputs);
}

void read_mnt4_g2(uint8_t *dest, FILE *inputs)
{
  read_mnt_fq2(dest, inputs);
  read_mnt_fq2(dest + 2 * bytes_per_elem, inputs);
}

void read_mnt6_g2(uint8_t *dest, FILE *inputs)
{
  read_mnt_fq3(dest, inputs);
  read_mnt_fq3(dest + 3 * bytes_per_elem, inputs);
}

void write_mnt_fq(uint8_t *fq, FILE *outputs)
{
  fwrite((void *)fq, io_bytes_per_elem * sizeof(uint8_t), 1, outputs);
}

void write_mnt_fq_montgomery(uint8_t *fq, FILE *outputs)
{
  Fq<mnt4753_pp> x;
  memcpy((void *)x.mont_repr.data, fq, io_bytes_per_elem);

  Fq<mnt4753_pp> result = Fq<mnt4753_pp>(x.mont_repr);
  fwrite((void *)result.mont_repr.data, io_bytes_per_elem * sizeof(uint8_t), 1, outputs);
}

void write_mnt_fq2(uint8_t *src, FILE *outputs)
{
  write_mnt_fq(src, outputs);
  write_mnt_fq(src + bytes_per_elem, outputs);
}

void write_mnt_fq3(uint8_t *src, FILE *outputs)
{
  write_mnt_fq(src, outputs);
  write_mnt_fq(src + bytes_per_elem, outputs);
  write_mnt_fq(src + 2 * bytes_per_elem, outputs);
}

void write_mnt_g1_montgomery(uint8_t *src, FILE *outputs)
{
  write_mnt_fq_montgomery(src, outputs);
  write_mnt_fq_montgomery(src + bytes_per_elem, outputs);
}

void write_mnt4_g2(uint8_t *src, FILE *outputs)
{
  write_mnt_fq2(src, outputs);
  write_mnt_fq2(src + 2 * bytes_per_elem, outputs);
}

void write_mnt6_g2(uint8_t *src, FILE *outputs)
{
  write_mnt_fq3(src, outputs);
  write_mnt_fq3(src + 3 * bytes_per_elem, outputs);
}

// LIBFF reading montgomery and numeral representations from array
Fq<mnt4753_pp> libff_read_mnt4_fq_numeral(uint8_t *src)
{
  // bigint<mnt4753_q_limbs> n;
  Fq<mnt4753_pp> x;
  memcpy((void *)x.mont_repr.data, src, libff::mnt4753_q_limbs * sizeof(mp_size_t));
  return Fq<mnt4753_pp>(x.mont_repr);
}

G1<mnt4753_pp> libff_read_mnt4_g1_numeral(uint8_t *src)
{
  Fq<mnt4753_pp> x = libff_read_mnt4_fq_numeral(src);
  Fq<mnt4753_pp> y = libff_read_mnt4_fq_numeral(src + bytes_per_elem);
  Fq<mnt4753_pp> z = libff_read_mnt4_fq_numeral(src + 2 * bytes_per_elem);
  return G1<mnt4753_pp>(x, y, z);
}

void init_libff()
{
  mnt4753_pp::init_public_params();
  mnt6753_pp::init_public_params();
}

uint8_t *g1_to_affine(uint8_t *src)
{
  G1<mnt4753_pp> g = libff_read_mnt4_g1_numeral(src);
  g.to_affine_coordinates();

  uint8_t *dst = new uint8_t[3 * bytes_per_elem];
  memset(dst, 0, 3 * bytes_per_elem);

  memcpy(dst, (uint8_t *)g.X().as_bigint().data, io_bytes_per_elem);
  memcpy(dst + bytes_per_elem, (uint8_t *)g.Y().as_bigint().data, io_bytes_per_elem);
  memcpy(dst + 2 * bytes_per_elem, (uint8_t *)g.Z().as_bigint().data, io_bytes_per_elem);

  return dst;
}
