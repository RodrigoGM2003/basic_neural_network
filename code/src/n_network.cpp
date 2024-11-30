#include "n_network.h"
#include <iostream>


n_network::n_network(int num_layers, int num_inputs, int num_outputs,
                     const activation& hidden_activation,
                     const activation& output_activation){
    layers = vector<layer>(num_layers);
    this->num_layers = num_layers;
    this->num_inputs = num_inputs;
    this->num_outputs = num_outputs;

   int i = 0;

   if(num_layers == 1) layers[i] = layer(num_outputs,num_inputs, output_activation);
   else layers[i] = layer(1,num_inputs, hidden_activation);

   for(i = 1; i < num_layers - 1; i++)
       layers[i] = layer(1,layers[i-1].get_nodes(), hidden_activation);

    layers.back() = layer(num_outputs, layers[i-1].get_nodes(), output_activation);
}


void n_network::set_hidden_function(const activation& new_activation) {
    for(int i = 0; i < num_layers - 1; i++)
        layers[i].set_activation_function(new_activation);
}

void n_network::set_output_function(const activation& new_activation) {
    layers.back().set_activation_function(new_activation);
}

void n_network::set_layer_function(int layer, const activation& new_activation) {
    if(layer >= 0 && layer < num_layers)
        layers[layer].set_activation_function(new_activation);
}

void n_network::set_layer_nodes(int layer, int nodes) {
    if(layer < 0 || layer >= num_layers || nodes < 1) return;

    layers[layer].set_nodes(nodes);

    if(layer == num_layers - 1) num_outputs = nodes;
    else{
        layers[layer + 1].set_inputs(nodes);
    }
}

void n_network::remove_layer() {
    if(num_layers == 1) return;

    vector<layer>::iterator it = layers.begin() + 1;

    (it+1)->set_inputs(layers.begin()->get_inputs());

    layers.erase(it);
}

void n_network::add_layer() {
    layers.emplace(layers.end() - 2);
}

void n_network::show_weights() {
    for(const layer& l : layers){
        l.show_weights();
        std::cout << std::endl;
    }
}
void n_network::randomize(){
    for(layer& l : layers)
        l.randomize();
}

void n_network::initialize_gradients() {
    for(layer& l : layers)
        l.initialize_gradient();
}

void n_network::free_gradients() {
    for(layer& l : layers)
        l.free_gradient();
}

vector<double> n_network::calculate_outputs(const vector<unsigned char>& input){
    vector<double> result;
    //Forward pass
    result = layers[0].calculate_outputs(input);
    for(int i = 1; i < num_layers; i++)
        result = layers[i].calculate_outputs(result);

    return result;
}
double n_network::cost(const vector<unsigned char>& input,
                       const vector<double>& expected_output){
    double cost = 0;

    //Forward pass
    vector<double> outputs = calculate_outputs(input);

    //Calculate cost
    for(int i = 0; i < num_outputs; i++)
        cost += layer::node_cost(outputs[i], expected_output[i]);

    return cost;
}
double n_network::cost(const data_set& dataset, int start_pos, int batch_size){
    double total_cost = 0;

    //For each element in the dataset calculate the cost
    for(int i = 0; i < batch_size; i++) {
        vector<double> aux(num_outputs, 0);
        aux[dataset.labels[i + start_pos]] = 1;

        total_cost += cost(dataset.data[i + start_pos], aux);
    }

    //Return the average cost
    return total_cost / batch_size;
}

void n_network::calculate_gradient(const vector<unsigned char>& input, const vector<double>& expected_output){
    //Forward pass
    calculate_outputs(input);

    //Calculate gradients of last layer
    layers[num_layers - 1].calculate_output_gradient(layers[num_layers - 2].get_outputs(), expected_output);

    //Calculate gradients of hidden layers
    for(int i = num_layers - 2; i > 0; i--)
        layers[i].calculate_hidden_gradient(layers[i - 1].get_outputs(), layers[i + 1]);

    //Calculate gradients of first layer
    layers[0].calculate_hidden_gradient(input, layers[1]);
}
void n_network::update_weights(int batch_size, double learning_rate) {
    //Update weights of each layer
    for(layer& l : layers)
        l.update_weights(batch_size, learning_rate);
}

void n_network::learn(const data_set& dataset, int batch_size, double learning_rate, int epochs){
    //Initialize gradients
    this->initialize_gradients();

    //Print initial cost
    std::cout << cost(dataset,0,100) << std::endl;

    //For each epoch
    for(int epoch = 0; epoch < epochs; epoch++){

        //For each element in the dataset
        for(int i = 0; i < dataset.data.size(); i++){
            
            //Forward pass till reaching the batch size
            vector<double> aux(this->num_outputs, 0);
            aux[dataset.labels[i]] = 1;

            //Update the gradients
            this->calculate_gradient(dataset.data[i], aux);

            //Update weights once a batch is reached
            if(i % batch_size == 0)
                this->update_weights(batch_size, learning_rate);

        }

        //Print the updated cost
        std::cout<<cost(dataset,0,100)<<std::endl;
    }

    //Free gradients
    this->free_gradients();
}


n_network& n_network::operator=(const n_network& other){
    if(this != &other){
        this->layers = other.layers;
        this->num_layers = other.num_layers;
        this->num_inputs = other.num_inputs;
        this->num_outputs = other.num_outputs;
    }

    return *this;
}
