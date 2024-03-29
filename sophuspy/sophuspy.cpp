// #include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include "python/root.h"
#include "python/so2.h"
#include "python/so3.h"
#include "python/se2.h"
#include "python/se3.h"

namespace Sophus
{
PYBIND11_MODULE(sophuspy, m)
{
	declareRoot(m);

	declareSO2(m);
	declareSE2(m);

	declareSO3(m);
	declareSE3(m);
}
} // end namespace Sophus