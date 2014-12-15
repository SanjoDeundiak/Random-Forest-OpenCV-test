// Pclass + Sex + Age + SibSp + Parch + Fare + Embarked + FamilySize

#include "Build.h"

int build_rtrees_classifier(char* data_filename, char* filename_to_save)
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

    // Create Random Trees classifier
    
    // create classifier by using <data> and <responses>
    printf("Training the classifier ...\n");

    // 1. create type mask
    var_type = cvCreateMat(data->cols + 1, 1, CV_8U);
    cvSet(var_type, cvScalarAll(CV_VAR_ORDERED));
    cvSetReal1D(var_type, data->cols, CV_VAR_CATEGORICAL);

    // 2. train classifier
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

    printf("\nTrain error %f\n", forest.get_train_error());

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