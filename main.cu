#include <cstdio>
#include <cstring>
#include <cassert>
#include <vector>

#include "fixnum/warp_fixnum.cu"
#include "array/fixnum_array.h"
#include "functions/modexp.cu"
#include "functions/multi_modexp.cu"
#include "modnum/modnum_monty_redc.cu"
#include "modnum/modnum_monty_cios.cu"

using namespace std;
using namespace cuFIXNUM;


const unsigned int bytes_per_elem = 128;
const unsigned int io_bytes_per_elem = 96;

// gpu modulus
__constant__ uint8_t fq_modulus[bytes_per_elem];

template <typename fixnum>
class GpuFq
{
  typedef modnum_monty_cios<fixnum> modnum;
private:
  fixnum data_;
  modnum& mod;
public:

  __device__
  GpuFq(const fixnum &data, modnum& mod) : data_(data), mod(mod) {}

  __device__ static GpuFq load(const fixnum &data, modnum& mod) 
  {
    fixnum result;
    mod.to_modnum(result, data);
    return GpuFq(result, mod);
  }

  __device__ __forceinline__ void save(fixnum& result) {
    this->mod.from_modnum(result, this->data_);
  }

  __device__ __forceinline__ GpuFq operator*(const GpuFq &other) const
  {
    fixnum result;
    this->mod.mul(result, this->data_, other.data_);
    return GpuFq(result, this->mod);
  }

  __device__ __forceinline__ GpuFq operator+(const GpuFq &other) const
  {
    fixnum result;
    this->mod.add(result, this->data_, other.data_);
    return GpuFq(result, this->mod);
  }

  __device__ __forceinline__ GpuFq operator-(const GpuFq &other) const
  {
    fixnum result;
    this->mod.sub(result, this->data_, other.data_);
    return GpuFq(result, this->mod);
  }

  __device__ __forceinline__ GpuFq squared() const
  {
    fixnum result;
    this->mod.sqr(result, this->data_);
    return GpuFq(result, this->mod);
  }

  __device__ __forceinline__ bool is_zero() const
  {
    return fixnum::is_zero(this->data_);
  }
};

template< typename fixnum >
struct mul_and_convert {
  // redc may be worth trying over cios
  typedef modnum_monty_cios<fixnum> modnum;
  typedef GpuFq<fixnum> GpuFq;

  __device__ void operator()(fixnum &r, fixnum a, fixnum b) {
      modnum mod = modnum(*(fixnum*)fq_modulus);

      GpuFq fqA = GpuFq::load(a, mod);
      GpuFq fqB = GpuFq::load(b, mod);
      GpuFq fqS = fqA * fqB;
      fqS.save(r);
  }
};

template< int fn_bytes, typename fixnum_array >
void print_fixnum_array(fixnum_array* res, int nelts) {
    int lrl = fn_bytes*nelts;
    uint8_t local_results[lrl];
    int ret_nelts;
    for (int i = 0; i < lrl; i++) {
      local_results[i] = 0;
    }
    res->retrieve_all(local_results, fn_bytes*nelts, &ret_nelts);

    for (int i = 0; i < lrl; i++) {
      printf("%i ", local_results[i]);
    }
    printf("\n");
}

template< int fn_bytes, typename fixnum_array >
uint8_t* get_fixnum_array(fixnum_array* res, int nelts) {
    int lrl = fn_bytes*nelts;
    uint8_t* local_results = new uint8_t[lrl];
    int ret_nelts;
    res->retrieve_all(local_results, fn_bytes*nelts, &ret_nelts);
    return local_results;
}


template< int fn_bytes, typename word_fixnum, template <typename> class Func >
uint8_t* compute_product(uint8_t* a, uint8_t* b, int nelts, uint8_t* input_m_base) {
    typedef warp_fixnum<fn_bytes, word_fixnum> fixnum;
    typedef fixnum_array<fixnum> fixnum_array;

    cudaMemcpyToSymbol(fq_modulus, input_m_base, bytes_per_elem);

    fixnum_array *res, *in_a, *in_b;
    in_a = fixnum_array::create(a, fn_bytes * nelts, fn_bytes);
    in_b = fixnum_array::create(b, fn_bytes * nelts, fn_bytes);
    res = fixnum_array::create(nelts);

    fixnum_array::template map<Func>(res, in_a, in_b);

    uint8_t* v_res = get_fixnum_array<fn_bytes, fixnum_array>(res, nelts);

    //TODO to do stage 1 field arithmetic, instead of a map, do a reduce

    delete in_a;
    delete in_b;
    delete res;
    return v_res;
}

void read_mnt_fq(uint8_t* dest, FILE* inputs) {
  // the input is montgomery representation x * 2^768 whereas cuda-fixnum expects x * 2^1024 so we shift over by (1024-768)/8 bytes
  fread((void*)(dest), io_bytes_per_elem*sizeof(uint8_t), 1, inputs);
}

void write_mnt_fq(uint8_t* fq, FILE* outputs) {
  fwrite((void *) fq, io_bytes_per_elem * sizeof(uint8_t), 1, outputs);
}

void print_array(uint8_t* a) {
  for (int j = 0; j < 128; j++) {
    printf("%x ", ((uint8_t*)(a))[j]);
  }
  printf("\n");
}

int main(int argc, char* argv[]) {
  setbuf(stdout, NULL);

  // mnt4_q
  uint8_t mnt4_modulus[bytes_per_elem] = {1,128,94,36,222,99,144,94,159,17,221,44,82,84,157,227,240,37,196,154,113,16,136,99,164,84,114,118,233,204,90,104,56,126,83,203,165,13,15,184,157,5,24,242,118,231,23,177,157,247,90,161,217,36,209,153,141,237,160,232,37,185,253,7,115,216,151,108,249,232,183,94,237,175,143,91,80,151,249,183,173,205,226,238,34,144,34,16,17,196,146,45,198,196,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};

  // mnt6_q
  uint8_t mnt6_modulus[bytes_per_elem] = {1,0,0,64,226,118,7,217,79,58,161,15,23,153,160,78,151,87,0,63,188,129,195,214,164,58,153,52,118,249,223,185,54,38,33,41,148,202,235,62,155,169,89,200,40,92,108,178,157,247,90,161,217,36,209,153,141,237,160,232,37,185,253,7,115,216,151,108,249,232,183,94,237,175,143,91,80,151,249,183,173,205,226,238,34,144,34,16,17,196,146,45,198,196,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};

  auto inputs = fopen(argv[2], "r");
  auto outputs = fopen(argv[3], "w");

  size_t n;

   while (true) {
    size_t elts_read = fread((void *) &n, sizeof(size_t), 1, inputs);
    if (elts_read == 0) { break; }

    uint8_t* x0 = new uint8_t[n * bytes_per_elem];
    for (size_t i = 0; i < n; ++i) {
      read_mnt_fq(x0 + i * bytes_per_elem, inputs);
    }

    uint8_t* x1 = new uint8_t[n * bytes_per_elem];
    for (size_t i = 0; i < n; ++i) {
      read_mnt_fq(x1 + i * bytes_per_elem, inputs);
    }

    uint8_t* res_x = compute_product<bytes_per_elem, u64_fixnum, mul_and_convert>(x0, x1, n, mnt4_modulus);

    for (size_t i = 0; i < n; ++i) {
      write_mnt_fq(res_x + i * bytes_per_elem, outputs);
    }

    uint8_t* y0 = new uint8_t[n * bytes_per_elem];
    for (size_t i = 0; i < n; ++i) {
      read_mnt_fq(y0 + i * bytes_per_elem, inputs);
    }

    uint8_t* y1 = new uint8_t[n * bytes_per_elem];
    for (size_t i = 0; i < n; ++i) {
      read_mnt_fq(y1 + i * bytes_per_elem, inputs);
    }

    uint8_t* res_y = compute_product<bytes_per_elem, u64_fixnum, mul_and_convert>(y0, y1, n, mnt6_modulus);

    for (size_t i = 0; i < n; ++i) {
      write_mnt_fq(res_y + i * bytes_per_elem, outputs);
    }

    delete x0;
    delete x1;
    delete y0;
    delete y1;
    delete res_x;
    delete res_y;
  }

  return 0;
}

