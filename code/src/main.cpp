#include <iostream>
#include <unistd.h>
#include <stdio.h>
#include <string.h>

#include "n_network.h"
#include "data_set.h"


int main(int argc, char * argv[]) {
        std::cout<<"AQUI "<<std::endl;

            std::cout.flush();  // Explicitly flush the output


    // srand((unsigned) time(NULL));

    // // string data_path = get_current_dir_name();
    // //data_path += "/data/train-images.idx3-ubyte";
    // string data_path = "../../data/t10k-images.idx3-ubyte";

    // // string label_path = get_current_dir_name();
    // //label_path += "/data/train-labels.idx1-ubyte";
    // string label_path = "../../data/t10k-labels.idx1-ubyte";

    // data_set d(data_path,label_path);

    // std::cout<<std::endl<<std::endl;

    // std::cout<<"AQUI 1"<<std::endl;
    // n_network network(3, 28*28, 10,
    //                   sig_activation, sig_activation);
    // network.set_layer_nodes(0,32);
    // network.set_layer_nodes(1,16);

    // std::cout<<"AQUI 2"<<std::endl;

    // network.learn(d,100, 1, 3);

    // d.close();

    // // data_path = get_current_dir_name();
    // data_path += "../../data/t10k-images.idx3-ubyte";

    // // label_path = get_current_dir_name();
    // label_path += "../../data/t10k-labels.idx1-ubyte";

    // std::cout<<"AQUI 3"<<std::endl;

    // d.open(data_path, label_path);

    // std::cout<<"AQUI 4"<<std::endl;

    // int total_hits = 0;
    // for(int i = 0; i < 100; i++) {
    //     for (int j = 0; j < 28 * 28; j++) {
    //         if (j % 28 == 0)
    //             std::cout << std::endl;
    //         std::cout << (int) d.data[i][j] << "\t";
    //     }
    //     std::cout << std::endl;
    //     std::cout << "The expected result is: " << (int) d.labels[i];

    //     auto aux = network.calculate_outputs(d.data[i]);
    //     double max = 0;
    //     int max_pos = 0;

    //     std::cout << std::endl;
    //     for (int j = 0; j < aux.size(); j++) {
    //         std::cout<<aux[j]<< "    ";
    //         if (aux[j] > max) {
    //             max = aux[j];
    //             max_pos = j;
    //         }
    //     }
    //     if(max_pos == d.labels[i]) total_hits++;
    //     std::cout<<std::endl;
    //     std::cout<<"The obtained result is: " <<max_pos<<" with a "<<max * 100<<"% of confidence";

    // }
    // std::cout<<std::endl;
    // std::cout<<"The total cost is: " <<network.cost(d, 0, d.data.size());
    // std::cout<<std::endl;
    // std::cout<<"Total hits: "<< total_hits;
}