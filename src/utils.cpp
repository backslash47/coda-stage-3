#include <cstdio>

#include "utils.hpp"
#include "constants.hpp"

void print_array(uint8_t *a)
{
  for (int j = 0; j < 128; j++)
  {
    printf("%x ", ((uint8_t *)(a))[j]);
  }
  printf("\n");
}

void printG1(uint8_t *src)
{
  printf("X:");
  print_array(src);
  printf("Y:");
  print_array(src + bytes_per_elem);
  printf("Z:");
  print_array(src + 2 * bytes_per_elem);
}
