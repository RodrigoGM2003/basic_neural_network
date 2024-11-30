#ifndef LAYER_H
#define LAYER_H

#include <vector>
#include <ctime>
#include <random>
#include  <iostream>

#include "functions.h"


using namespace std;

/**
 * @brief Class that represents a layer of a neural network
 */
class layer {
private:
    vector<vector<double>> weights; //*< Weights of the layer */
    vector<double> bias; //*< Bias of the layer */

    vector<double> outputs; //*< Outputs of the layer */
    vector<double> deltas; //*< Deltas of the layer */

    vector<vector<double>> weight_gradients; //*< Gradients of the weights */
    vector<double> bias_gradients; //*< Gradients of the bias */
 
    activation activation_function; //*< Activation function of the layer */
    int nodes, inputs; //*< Number of nodes and inputs of the layer */


public:
    
    /**
     * @brief Constructor
     * @param nodes Number of nodes of the layer
     * @param inputs Number of inputs of the layer
     * @param activation_function Activation function of the layer (default ReLu)
     */
    explicit layer(int nodes = 1, int inputs = 1, const activation& activation_function = ReLu_activation);

    /**
     * @brief Copy constructor
     * @param other Other layer
     */
    layer(const layer& other);

    /**
     * @brief Destructor
     */
    ~layer() = default;

    /**
     * @brief Get the bias of a node
     * @param node Node
     * @return Bias of the node
     */
    [[nodiscard]] inline double get_bias(int node) const {return bias[node];};

    /**
     * @brief Get the weight of a node
     * @param node Node
     * @param input Input
     * @return Weight of the node
     */
    [[nodiscard]] inline double get_weight(int node, int input) const {return this->weights[node][input];};

    /**
     * @brief Get the output of a node
     * @param node Node
     * @return Output of the node
     */
    [[nodiscard]] inline double get_output(int node) const {return outputs[node];};

    /**
     * @brief Get the delta of a node
     * @param node Node
     * @return Delta of the node (used for backpropagation)
     */
    [[nodiscard]] inline double get_delta(int node) const {return deltas[node];};

    /**
     * @brief Get the number of nodes of the layer
     * @return Number of nodes of the layer
     */
    [[nodiscard]] inline int get_nodes() const {return nodes;};

    /**
     * @brief Get the number of inputs of the layer
     * @return Number of inputs of the layer
     */
    [[nodiscard]] inline int get_inputs() const {return inputs;};

    /**
     * @brief Get the activation function of the layer
     * @return Activation function of the layer
     */
    [[nodiscard]] inline const activation& get_activation_function() const {return activation_function;};

    /**
     * @brief Get the outputs of the layer
     */
    [[nodiscard]] inline const vector<double>& get_outputs() const {return outputs;};

    /**
     * @brief Get the deltas of the layer
     */
    [[nodiscard]] inline const vector<double>& get_deltas() const {return deltas;};

    
    /**
     * @brief Set the activation function of the layer
     */
    void set_activation_function(const activation& new_activation);

    /**
     * @brief Set the number of nodes of the layer
     */
    void set_nodes(int num_nodes);

    /**
     * @brief Set the number of inputs of the layer
     */
    void set_inputs(int num_inputs);

    /**
     * @brief Add a node to the layer
     */
    void add_node();

    /**
     * @brief Remove a node from the layer
     */
    void remove_node();

    /**
     * @brief Add an input to the layer
     */
    void add_input();

    /**
     * @brief Remove an input from the layer
     */
    void remove_input();

    /**
     * @brief Print the weights of the layer
     * @details Used for debugging
     */
    void show_weights() const ;

    /**
     * @brief Randomize the weights and bias of the layer
     */
    void randomize();

    /**
     * @brief Calculate the outputs of the layer (Forward pass)
     * @param input_vector Input vector
     * @return Outputs of the layer
     */
    const vector<double>& calculate_outputs(const vector<unsigned char>& input_vector);

    /**
     * @brief Calculate the outputs of the layer (Forward pass)
     * @param input_vector Input vector
     * @return Outputs of the layer
     */
    const vector<double>& calculate_outputs(const vector<double>& input_vector);

    /**
     * @brief Calculate the gradient of the output layer (Backpropagation)
     * @param input Input vector
     * @param expected_outputs Expected outputs
     */
    void calculate_output_gradient(const vector<double>& input,
                                   const vector<double>& expected_outputs);    

    /**
     * @brief Calculate the gradient of the output layer (Backpropagation)
     * @param input Input vector
     * @param expected_outputs Expected outputs
     */                              
    void calculate_output_gradient(const vector<unsigned char>& input,
                                   const vector<double>& expected_outputs);

    /**
     * @brief Calculate the gradient of a hidden layer (Backpropagation)
     * @param input Input vector
     * @param previous_layer Previous layer
     */
    void calculate_hidden_gradient(const vector<double>& input,
                                   const layer& previous_layer);

    /**
     * @brief Calculate the gradient of a hidden layer (Backpropagation)
     * @param input Input vector
     * @param previous_layer Previous layer
     */
    void calculate_hidden_gradient(const vector<unsigned char>& input,
                                   const layer& previous_layer);

    /**
     * @brief Update the weights of the layer (Backpropagation)
     * @param batch_size Size of the batch
     * @param learning_rate Learning rate
     */
    void update_weights(int batch_size, double learning_rate);

    /**
     * @brief Initialize the gradient of the layer
     */
    void initialize_gradient();

    /**
     * @brief Free the gradient of the layer
     */
    void free_gradient();

    /**
     * @brief Calculate the cost of a node
     * @param output Output of the node
     * @param expected_output Expected output of the node
     * @return Cost of the node
     * @details Mean squared error
     */
    static double node_cost(double output, double expected_output) ;

    /**
     * @brief Calculate the derivative of the cost of a node
     * @param output Output of the node
     * @param expected_output Expected output of the node
     * @return Derivative of the cost of the node
     */
    static double d_node_cost(double output, double expected_output) ;

    /**
     * @brief Copy operator
     * @param other Other layer
     * @return Copied layer
     */
    layer& operator=(const layer& other);

    
private:

    /**
     * @brief Random double between 0 and 1
     * @return Random double
     */
    static double random_double();
};


#endif 
