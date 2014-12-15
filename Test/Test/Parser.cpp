#include "Parser.h"

#include <iostream>
#include <fstream>

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
        float res = -1.;
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

    throw - 1;
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