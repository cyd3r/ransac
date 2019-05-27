#include <cmath>

int triangMax(int n)
{
    return (n * (n - 1)) / 2;
}

int triang_tuple(int x, int y)
{
    return (y * (y - 1)) / 2 + x;
}

void tuple_triang(int t, int &x, int &y)
{
    y = (int)((1 + (int)sqrt(1 + 8 * t)) / 2);
    x = t - (y * (y - 1)) / 2;
}