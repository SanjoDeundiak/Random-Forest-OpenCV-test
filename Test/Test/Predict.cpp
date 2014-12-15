#include "Parser.h"
#include "Predict.h"

#include <fstream>

int predict_rtrees(char* data_filename,
    char* filename_to_load, char* results)
{
    CvMat* data = 0;
    CvMat* responses = 0;

    int ok = read_num_class_data(data_filename, 9, &data, &responses);

    CvRTrees forest;
    int nsamples = data->rows;
    if (filename_to_load)
    {
        // load classifier from the specified file
        forest.load(filename_to_load);
        if (forest.get_tree_count() == 0)
        {
            printf("Could not read the classifier %s\n", filename_to_load);
            return -1;
        }
        printf("The classifier %s is loaded.\n", filename_to_load);
    }

    int startId = 892;
    std::ofstream res(results);
    res << "PassengerId,Survived" << std::endl;
    for (int i = 0; i < nsamples; i++)
    {
        CvMat sample;
        cvGetRow(data, &sample, i);

        res << startId + i << ',' << forest.predict(&sample) << std::endl;
    }

    printf("Done\n");
    return 0;
}