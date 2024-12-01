#include <iostream>
#include <unistd.h>
#include <stdio.h>
#include <string.h>

#include "n_network.h"
#include "data_set.h"


int main(int argc, char * argv[]) {
        
    srand((unsigned) time(NULL));

    //Open the dataset
    string data_path = "../../data/train-images.idx3-ubyte";
    string label_path = "../../data/train-labels.idx1-ubyte";
    data_set d(data_path, label_path);

    std::cout<<std::endl;

    //Create the network with 3 layers, 28*28 inputs and 10 outputs
    n_network network(3, 28*28, 10,
                      sig_activation, sig_activation);
    network.set_layer_nodes(0,32);
    network.set_layer_nodes(1,16);

    //Initialize the hyperparameters
    int batch_size = 10;
    float learning_rate = 1;
    int epochs = 10;

    //Train the network
    network.learn(d, batch_size, learning_rate, epochs);

    //Close the dataset
    d.close();

    //Open the test dataset with the test images
    data_path = "../../data/t10k-images.idx3-ubyte";
    label_path = "../../data/t10k-labels.idx1-ubyte";
    d.open(data_path, label_path);

    //Test the network
    int total_hits = 0;
    for(int i = 0; i < 100; i++) {
        //Calculate the output of the network (Forward pass)
        auto aux = network.calculate_outputs(d.data[i]);

        //Get the maximum value of the output (the predicted label)
        double max = 0;
        int max_pos = 0;
        for (int j = 0; j < aux.size(); j++) {
            if (aux[j] > max) {
                max = aux[j];
                max_pos = j;
            }
        }
        
        //Check if the predicted label is correct
        if(max_pos == d.labels[i]) total_hits++;
    }

    d.close();

    //Show the results
    std::cout<<std::endl;
    std::cout<<"The total cost is: " <<network.cost(d, 0, d.data.size());
    std::cout<<std::endl;
    std::cout<<"Total accuracy: "<< total_hits << "%";
}