//
// Created by rodrigogm on 7/02/23.
//

#include "layer.h"
using namespace std;


layer::layer(int nodes, int inputs, const activation& activation_function) {
    this->nodes = nodes;
    this->inputs = inputs;

    this->bias = vector<double>(this->nodes,0.01);

    this->outputs = vector<double>(this->nodes);
    this->deltas = vector<double>(this->nodes);

    this->activation_function = activation_function;

    //Initialize weights with random values
    weights.resize(this->nodes);
    for(vector<double>& node : weights){
        node.resize(inputs);
        for(double& w : node)
            w = random_double();
    }
}
layer::layer(const layer& other) {
    *this = other;
}

void layer::set_activation_function(const activation& new_activation){
    this->activation_function = new_activation;
}

void layer::set_nodes(int num_nodes){
    layer aux(num_nodes, this->inputs, activation_function);

    *this = aux;
}
void layer::set_inputs(int num_inputs){
    layer aux(this->nodes, num_inputs, activation_function);

    *this = aux;
}
void layer::add_node(){
    weights.emplace_back(inputs);

    for(double& w : weights.back())
        w = random_double();

    nodes++;
}
void layer::remove_node(){
    if(nodes > 0) {
        weights.pop_back();
        nodes--;
    }
}
void layer::add_input(){
    for(vector<double>& node : weights)
        node.push_back(random_double());

    inputs++;
}
void layer::remove_input(){
    if(inputs != 0) {
        for (vector<double>& node: weights)
            node.pop_back();
        inputs--;
    }
}

void layer::show_weights() const {
    for(const vector<double>& v : weights) {
        for (double d: v)
            cout << d << " ";
        cout<<endl;
    }
}

void layer::randomize() {
    //Randomize all weights
    for(vector<double>& v : weights)
        for(double& d : v)
            d = random_double();
}

const vector<double>& layer::calculate_outputs (const vector<unsigned char>& input_vector){
    //Convert input vector to double
    vector<double> aux = vector<double>(input_vector.size());

    //Copy values (unefficient)
    for(int i = 0; i < aux.size(); i++)
        aux[i] = input_vector[i];

    //Forward pass
    calculate_outputs(aux);

    return outputs;
}

const vector<double>& layer::calculate_outputs (const vector<double>& input_vector) {
    for (int node = 0; node < this->nodes; node++) {
        //The bias is added
        outputs[node] = bias[node];

        //To the output of the node, the weighted sum of the inputs is added
        for (int i = 0; i < this->inputs; i++)
            outputs[node] += input_vector[i] * weights[node][i];

        //Then the activation function is applied
        outputs[node] = activation_function.function(outputs[node]);
    }

    return outputs;
}

void layer::calculate_output_gradient(const vector<double>& input,
                                      const vector<double>& expected_outputs){
    for(int i = 0; i < this->nodes; i++){
        //Calculate the delta of the node (deltas are used in backpropagation, chain rule)
        this->deltas[i] = d_node_cost(outputs[i], expected_outputs[i]) *
                            activation_function.derivative(outputs[i]);
        
        //Calculate the gradient of the bias and the weights
        this->bias_gradients[i] += this->deltas[i];

        //For each weight, the gradient is calculated
        for(int j = 0; j < this->inputs; j++)
            this->weight_gradients[i][j] += this->deltas[i] * input[j];
    }
}

void layer::calculate_output_gradient(const vector<unsigned char>& input,
                                      const vector<double>& expected_outputs){
    //Convert input vector to double
    vector<double> aux = vector<double>(input.size());

    //Copy values (unefficient)
    for(int i = 0; i < aux.size(); i++)
        aux[i] = input[i];

    //Call the double version of the function
    calculate_output_gradient(aux,expected_outputs);
}

void layer::calculate_hidden_gradient(const vector<double>& input,
                                      const layer& previous_layer){

    //For each node in the layer
    for(int i = 0; i < nodes; i++){
        //Initialize delta to 0
        double aux = 0;
        this->deltas[i] = 0;

        //For each node in the previous layer
        for(int j = 0; j < previous_layer.nodes; j++)
            //Add the delta of the previous layer node multiplied by the weight of the connection
            aux += previous_layer.deltas[j] * previous_layer.weights[j][i];

        //Multiply the sum by the derivative of the activation function and store it in the delta of the node
        this->deltas[i] += aux * activation_function.derivative(outputs[i]);

        //Calculate the gradient of the bias and the weights
        this->bias_gradients[i] += deltas[i];

        //For each weight, the gradient is calculated using the delta
        for(int j = 0; j < this->inputs; j++)
            this->weight_gradients[i][j] += this->deltas[i] * input[j];
    }
}

void layer::calculate_hidden_gradient(const vector<unsigned char>& input,
                                      const layer& previous_layer){
    //Convert input vector to double
    vector<double> aux = vector<double>(input.size());

    //Copy values (unefficient)
    for(int i = 0; i < aux.size(); i++)
        aux[i] = input[i];

    //Call the double version of the function
    calculate_hidden_gradient(aux, previous_layer);
}

void layer::update_weights(int batch_size, double learning_rate){
    for(int i = 0; i < nodes; i++){
        //Update the bias
        this->bias[i] -= learning_rate * (this->bias_gradients[i] / batch_size);
        this->bias_gradients[i] = 0;

        //Update the weights
        for(int j = 0; j < inputs; j++) {
            this->weights[i][j] -= learning_rate * (this->weight_gradients[i][j] / batch_size);
            this->weight_gradients[i][j] = 0;
        }
    }
}

void layer::initialize_gradient() {
    this->bias_gradients = vector<double>(this->nodes);

    this->weight_gradients.resize(this->nodes);
    for(vector<double>& node : weight_gradients)
        node = vector<double>(inputs,0);
}

void layer::free_gradient() {
    this->bias_gradients = {};
    this->weight_gradients = {};
}

double layer::random_double() {
    return (((double) rand()) / ((double) RAND_MAX) - 0.5) * 2;
}

double layer::node_cost(double output, double expected_output) {
    //Mean squared error
    double error = output - expected_output;
    return error * error;
}

double layer::d_node_cost(double output, double expected_output) {
    return 2 * (output - expected_output);
}

layer& layer::operator=(const layer& other) {
    if(this != &other){
        this->weights = other.weights;
        this->bias = other.bias;
        this->nodes = other.nodes;
        this->inputs = other.inputs;
        this->activation_function = other.activation_function;
        this->outputs = other.outputs;
        this->deltas = other.deltas;
        this->weight_gradients = other.weight_gradients;
        this->bias_gradients = other.bias_gradients;
    }

    return *this;
}
