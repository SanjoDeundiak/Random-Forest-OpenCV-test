#include "opencv2/core/core_c.h"
#include "opencv2/ml/ml.hpp"

#include <iostream>
#include <fstream>

#include "Build.h"
#include "Predict.h"

int main(int argc, char *argv[])
{
    char data_filename[] = "./newtrain.csv";
    char save_model[] = "random_forest.dat";
    char* load_model = save_model;
    char test_filename[] = "testt.csv";
    char results[] = "result.csv";

    build_rtrees_classifier(data_filename, save_model);
    predict_rtrees(test_filename, load_model, results);

    std::cin.ignore();
    return 0;
}