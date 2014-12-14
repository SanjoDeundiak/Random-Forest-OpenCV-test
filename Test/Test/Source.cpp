#include "opencv2/core/core_c.h"
#include "opencv2/ml/ml.hpp"

// This function reads data and responses from the file <filename>
int read_num_class_data(const char* filename, int var_count,
    CvMat** data, CvMat** responses)
{
    const int M = 1024;
    FILE* f = fopen(filename, "rt");
    CvMemStorage* storage;
    CvSeq* seq;
    char buf[M + 2];
    float* el_ptr;
    CvSeqReader reader;
    int i, j;

    if (!f)
        return 0;

    el_ptr = new float[var_count + 1];
    storage = cvCreateMemStorage();
    seq = cvCreateSeq(0, sizeof(*seq), (var_count + 1)*sizeof(float), storage);

    for (;;)
    {
        char* ptr;
        if (!fgets(buf, M, f) || !strchr(buf, ','))
            break;
        el_ptr[0] = buf[0];
        ptr = buf + 2;
        for (i = 1; i <= var_count; i++)
        {
            int n = 0;
            sscanf(ptr, "%f%n", el_ptr + i, &n);
            ptr += n + 1;
        }
        if (i <= var_count)
            break;
        cvSeqPush(seq, el_ptr);
    }
    fclose(f);

    *data = cvCreateMat(seq->total, var_count, CV_32F);
    *responses = cvCreateMat(seq->total, 1, CV_32F);

    cvStartReadSeq(seq, &reader);

    for (i = 0; i < seq->total; i++)
    {
        const float* sdata = (float*)reader.ptr + 1;
        float* ddata = data[0]->data.fl + var_count*i;
        float* dr = responses[0]->data.fl + i;

        for (j = 0; j < var_count; j++)
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

    int ok = read_num_class_data(data_filename, 16, &data, &responses);
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
    ntrain_samples = (int)(nsamples_all*0.8);

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

        // 2. create sample_idx
        sample_idx = cvCreateMat(1, nsamples_all, CV_8UC1);
        {
            CvMat mat;
            cvGetCols(sample_idx, &mat, 0, ntrain_samples);
            cvSet(&mat, cvRealScalar(1));

            cvGetCols(sample_idx, &mat, ntrain_samples, nsamples_all);
            cvSetZero(&mat);
        }

        // 3. train classifier
        forest.train(data, // inputr data
            CV_ROW_SAMPLE, // data is in rows
            responses, // respones
            0, // all features are important
            sample_idx, // ??
            var_type, // indicates that we're solving classification problem
            0, // missing data mask
            CvRTParams(10, // max depth
            10, // min sample count
            0, // regression accuracy
            false, // use surrogates ??
            15, // 
            0, true, 4, 100, 0.01f, CV_TERMCRIT_ITER));
        printf("\n");
    }

    // compute prediction error on train and test data
    for (i = 0; i < nsamples_all; i++)
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
        train_hr*100., test_hr*100.);

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

    //Print some proximitites
    printf("Proximities between some samples corresponding to the letter 'T':\n");
    {
        CvMat sample1, sample2;
        const int pairs[][2] = { { 0, 103 }, { 0, 106 }, { 106, 103 }, { -1, -1 } };

        for (i = 0; pairs[i][0] >= 0; i++)
        {
            cvGetRow(data, &sample1, pairs[i][0]);
            cvGetRow(data, &sample2, pairs[i][1]);
            printf("proximity(%d,%d) = %.1f%%\n", pairs[i][0], pairs[i][1],
                forest.get_proximity(&sample1, &sample2)*100.);
        }
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
    char default_data_filename[] = "./letter-recognition.data";
    char* data_filename = default_data_filename;

    build_rtrees_classifier(data_filename, filename_to_save, filename_to_load);
    return 0;
}
