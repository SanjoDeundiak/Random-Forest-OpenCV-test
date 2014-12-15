#include "opencv2/core/core_c.h"
#include "opencv2/ml/ml.hpp"

#include <iostream>
#include <fstream>

// Pclass + Sex + Age + SibSp + Parch + Fare + Embarked + FamilySize

const int numVar = 9;
// false = numerical, true - string
static bool isString[17] = { false, false, false, true, true, false, false, false, true, false,
                             true, true, true, false, true, true, true };
// true - not used
static bool ignore[17] = { true, false, false, true, false, false, false, false, true, false,
                           true, false, true, false, true, true, true };

float Convert(const std::string& val, int i)
{
    if (!isString[i])
    {
        std::stringstream ss(val);
        float res;
        ss >> res;
        return res;
    }
    
    if (val == "\"male\"")
        return 0.;
    if (val == "\"female\"")
        return 1.;
    if (val == "\"C\"")
        return 0.;
    if (val == "\"Q\"")
        return 1.;
    if (val == "\"S\"")
        return 2.;

    throw -1;
}

// This function reads data and responses from the file <filename>
int read_num_class_data(const char* filename, int var_count,
    CvMat** data, CvMat** responses)
{
    std::ifstream f(filename);
    CvMemStorage* storage;
    CvSeq* seq;
    float* el_ptr;
    CvSeqReader reader;

    if (!f)
        return 0;

    el_ptr = new float[var_count + 1];
    storage = cvCreateMemStorage();
    seq = cvCreateSeq(0, sizeof(*seq), (var_count + 1)*sizeof(float), storage);

    std::string line, value, temp;

    while (f)
    {
        int j = 0;
        line.clear();
        std::getline(f, line);
        if (line.empty())
            break;

        std::stringstream ss(std::move(line));
        for (int i = 0; std::getline(ss, value, ','); i++)
        {
            if (isString[i])
            while (value[value.size() - 1] != '"')
            {
                std::getline(ss, temp, ',');
                value += temp;
            }
            if (ignore[i])
                continue;

            el_ptr[j++] = Convert(value, i);
        }
        cvSeqPush(seq, el_ptr);
    }

    *data = cvCreateMat(seq->total, var_count, CV_32F);
    *responses = cvCreateMat(seq->total, 1, CV_32F);

    cvStartReadSeq(seq, &reader);

    for (int i = 0; i < seq->total; i++)
    {
        const float* sdata = (float*)reader.ptr + 1;
        float* ddata = data[0]->data.fl + var_count*i;
        float* dr = responses[0]->data.fl + i;

        for (int j = 0; j < var_count; j++)
            ddata[j] = sdata[j];
        *dr = sdata[-1];
        CV_NEXT_SEQ_ELEM(seq->elem_size, reader);
    }

    cvReleaseMemStorage(&storage);
    delete[] el_ptr;
    return 1;
}

int build_rtrees_classifier(char* data_filename,
    char* filename_to_save, char* filename_to_load)
{
    CvMat* data = 0;
    CvMat* responses = 0;
    CvMat* var_type = 0;
    CvMat* sample_idx = 0;

    int ok = read_num_class_data(data_filename, 9, &data, &responses);
    int nsamples_all = 0, ntrain_samples = 0;
    int i = 0;
    double train_hr = 0, test_hr = 0;
    CvRTrees forest;
    CvMat* var_importance = 0;

    if (!ok)
    {
        printf("Could not read the database %s\n", data_filename);
        return -1;
    }

    printf("The database %s is loaded.\n", data_filename);
    nsamples_all = data->rows;
    //ntrain_samples = (int)(nsamples_all*0.8);
    ntrain_samples = nsamples_all;

    // Create or load Random Trees classifier
    if (filename_to_load)
    {
        // load classifier from the specified file
        forest.load(filename_to_load);
        ntrain_samples = 0;
        if (forest.get_tree_count() == 0)
        {
            printf("Could not read the classifier %s\n", filename_to_load);
            return -1;
        }
        printf("The classifier %s is loaded.\n", filename_to_load);
    }
    else
    {
        // create classifier by using <data> and <responses>
        printf("Training the classifier ...\n");

        // 1. create type mask
        var_type = cvCreateMat(data->cols + 1, 1, CV_8U);
        cvSet(var_type, cvScalarAll(CV_VAR_ORDERED));
        cvSetReal1D(var_type, data->cols, CV_VAR_CATEGORICAL);

        // 3. train classifier
        forest.train(data, // input data
            CV_ROW_SAMPLE, // data is in rows
            responses, // respones
            0, // all features are important
            0,//sample_idx, // ??
            var_type, // indicates that we're solving classification problem
            0, // missing data mask
            CvRTParams
            (10, // max depth
            10, // min sample count
            0, // regression accuracy
            false, // use surrogates ??
            2, // max categories, used for clustering > 2 
            0, // prior class probabilities
            true, // calculate var importance
            3, // size of feature subset ~sqrt(8)
            150, // max num of trees in the forest
            0.01f, // accuracy
            CV_TERMCRIT_ITER  // termination type CV_TERMCRIT_ITER, CV_TERMCRIT_EPS, CV_TERMCRIT_ITER | CV_TERMCRIT_EPS
            ) // CvRTParams
            ); // train
        printf("\n");
    }

    printf("Train error %f\n", forest.get_train_error());

    // compute prediction error on train and test data
    /*for (i = 0; i < nsamples_all; i++)
    {
        double r;
        CvMat sample;
        cvGetRow(data, &sample, i);

        r = forest.predict(&sample);
        r = fabs((double)r - responses->data.fl[i]) <= FLT_EPSILON ? 1 : 0;

        if (i < ntrain_samples)
            train_hr += r;
        else
            test_hr += r;
    }

    test_hr /= (double)(nsamples_all - ntrain_samples);
    train_hr /= (double)ntrain_samples;
    printf("Recognition rate: train = %.1f%%, test = %.1f%%\n",
        train_hr*100., test_hr*100.);*/

    printf("Number of trees: %d\n", forest.get_tree_count());

    // Print variable importance
    var_importance = (CvMat*)forest.get_var_importance();
    if (var_importance)
    {
        double rt_imp_sum = cvSum(var_importance).val[0];
        printf("var#\timportance (in %%):\n");
        for (i = 0; i < var_importance->cols; i++)
            printf("%-2d\t%-4.1f\n", i,
            100.f*var_importance->data.fl[i] / rt_imp_sum);
    }

    // Save Random Trees classifier to file if needed
    if (filename_to_save)
        forest.save(filename_to_save);

    cvReleaseMat(&sample_idx);
    cvReleaseMat(&var_type);
    cvReleaseMat(&data);
    cvReleaseMat(&responses);

    return 0;
}

int main(int argc, char *argv[])
{
    char* filename_to_save = 0;
    char* filename_to_load = 0;
    char data_filename[] = "./newtrain.csv";
    //char data_filename[] = "./letter-recognition.data";
    //char data_filename[] = "./test.csv";

    build_rtrees_classifier(data_filename, filename_to_save, filename_to_load);

    std::cin.ignore();
    return 0;
}