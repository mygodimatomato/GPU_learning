#include <cstdio>
#include <tuple>
#include <array>

struct wrapper
{
  int *ptr;

  void operator=(int val) {
    *ptr = val /2 ;
  }
};

struct transform_output_iterator
{
  int *a;

  wrapper operator[](int i)
  {
    return wrapper{a + i};
  }
};

int main()
{
  std::array<int, 3> a{0,1,2};
  transform_output_iterator it {a.begin()};

  it[0] = 10;
  it[1] = 20;

  std::printf("a[0]: %d\n", a[0]);
  std::printf("a[1]: %d\n", a[1]);
}