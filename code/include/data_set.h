#ifndef DATA_SET_H
#define DATA_SET_H

#include <string>
#include <fstream>
#include <iostream>
#include <vector>
#include <map>

#include "functions.h"

using namespace std;

/**
 * @brief Struct that holds the data and labels of a dataset
 */
struct data_set { //Why a struct? I dont know, i was stupid back then
public:
    vector<vector<unsigned char>> data; //*< Data of the dataset */
    vector<unsigned char> labels; //*< Labels of the dataset */
    string path; //*< Path of the dataset */

    /**
     * @brief Constructor
     * @param data_path Path to the data file
     * @param label_path Path to the label file
     */
    explicit data_set(const string& data_path, const string& label_path); //*< Constructor */

    /**
     * @brief Destructor
     */
    ~data_set(){
        close();
    }

    /**
     * @brief Open the dataset
     * @param data_path Path to the data file
     * @param label_path Path to the label file
     */
    void open(const string& data_path, const string& label_path);

    /**
     * @brief Close the dataset
     */
    void close(){data = {}; labels = {};};
};


#endif
