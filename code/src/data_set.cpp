//
// Created by rodrigogm on 10/02/23.
//

#include "data_set.h"

data_set::data_set(const string& data_path, const string& label_path) {
    open(data_path, label_path);
}

void data_set::open(const string& data_path, const string& label_path){

    //OPENING AND READING MAGIC NUMBER AND DESCRIPTORS OF THE DATA
    ifstream fi_data(data_path,ios::binary);

    if (!fi_data.is_open()) {
        throw runtime_error("Could not open the data file: " + data_path);
    }
    int data_magic = 0, num_images = 0, num_rows = 0, num_cols = 0;

    fi_data.read((char *) &data_magic, sizeof(int));

    data_magic = reverseInt(data_magic);

    if(data_magic != 2051) throw runtime_error("Invalid MNIST image file!");

    fi_data.read((char *) &num_images, sizeof(int));
    num_images = reverseInt(num_images);

    fi_data.read((char *) &num_rows, sizeof(int));
    num_rows = reverseInt(num_rows);

    fi_data.read((char *) &num_cols, sizeof(int));
    num_cols = reverseInt(num_cols);

    //OPENING AND READING MAGIC NUMBER AND DESCRIPTORS OF LABELS
    ifstream fi_labels(label_path, ios::binary);
    int label_magic = 0, num_labels = 0;

    fi_labels.read((char *) &label_magic, sizeof(int));
    label_magic = reverseInt(label_magic);

    if(label_magic != 2049) throw runtime_error("Invalid MNIST label file!");

    fi_labels.read((char *) &num_labels, sizeof(int));
    num_labels = reverseInt(num_labels);

    //CHECKING FOR ERRORS
    if(num_images != num_labels)throw runtime_error("Number of labels not corresponding with number of images!");

    std::cout << "Dataset format: " << num_images << " images of size " << num_rows << "x" << num_cols << std::endl;
    std::cout << "Dataset labels: " << num_labels << std::endl;

    //READING IMAGES
    data.resize(num_images);
    for(int j = 0; j < num_images; j++) {
        data[j].resize(num_cols * num_rows);
        fi_data.read((char*)data[j].data(), num_rows*num_cols );
    }

    //READING LABELS
    labels.resize(num_labels);
    fi_labels.read((char*)labels.data(), num_labels);

    fi_data.close();
    fi_labels.close();
}

