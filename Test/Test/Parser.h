#pragma once

#include "opencv2/core/core_c.h"
#include "opencv2/ml/ml.hpp"

int read_num_class_data(const char* filename, int var_count,
    CvMat** data, CvMat** responses);