#ifndef N_NETWORK_H
#define N_NETWORK_H

#include <fstream>
#include <vector>

#include "layer.h"
#include "data_set.h"

/**
 * @brief Class that represents a neural network
 */
class n_network {
private:
    vector<layer> layers; //*< Layers of the network */
    int num_layers, num_inputs, num_outputs; //*< Number of layers, inputs and outputs of the network */
 
public:
    /**
     * @brief Constructor
     * @param num_layers Number of layers
     * @param num_inputs Number of inputs
     * @param num_outputs Number of outputs
     * @param hidden_activation Activation function of the hidden layers (default ReLu)
     * @param output_activation Activation function of the output layer (default Sigmoid)
     */
    explicit n_network( int num_layers = 1, int num_inputs = 1, int num_outputs = 1,
                        const activation& hidden_activation = ReLu_activation,
                        const activation& output_activation  = sig_activation);

    /**
     * @brief Copy constructor
     * @param other Other neural network
     */
    n_network(const n_network& other) = default;

    /**
     * @brief Destructor
     */
    ~n_network() = default;

    /**
     * @brief Get the number of layers of the network
     * @return Number of layers of the network
     */
    [[nodiscard]] int get_num_layers() const {return num_layers;};

    /**
     * @brief Get the input size of the network
     * @return Input size of the network
     */
    [[nodiscard]] int get_num_inputs() const {return num_inputs;};

    /**
     * @brief Get the size of the output of the network
     * @return Size of the output of the network
     */
    [[nodiscard]] int get_num_outputs() const {return num_outputs;};

    /**
     * @brief Get the layer of the network
     * @param layer Index of the layer
     * @return Layer of the network
     */
    [[nodiscard]] const layer& get_layer(int layer) const {return layers[layer];};

    /**
     * @brief Set the activation function of the hidden layers
     * @param new_activation New activation function
     */
    void set_hidden_function(const activation& new_activation);

    /**
     * @brief Set the activation function of the output layer
     * @param new_activation New activation function
     */
    void set_output_function(const activation& new_activation);

    /**
     * @brief Set the activation function of a layer
     * @param layer Index of the layer
     * @param new_activation New activation function
     */
    void set_layer_function(int layer, const activation& new_activation);

    /**
     * @brief Set the number of nodes of a layer
     * @param layer Index of the layer
     * @param nodes Number of nodes
     */
    void set_layer_nodes(int layer, int nodes);

    /**
     * @brief Remove a layer from the network
     */
    void remove_layer();

    /**
     * @brief Add a layer to the network
     */
    void add_layer();

    /**
     * @brief Show the weights of the network
     * @details This function is used for debugging purposes
     */
    void show_weights();

    /**
     * @brief Randomize the weights of the network
     */
    void randomize();

    /**
     * @brief Initialize the gradients of the network
     */
    void initialize_gradients();

    /**
     * @brief Free the gradients of the network
     */
    void free_gradients();

    /**
     * @brief Calculate the outputs of the network (Forward pass)
     * @param input Input vector
     * @return Output vector
     */
    vector<double> calculate_outputs(const vector<unsigned char>& input);

    /**
     * @brief Calculate the cost of an input
     * @param input Input vector
     * @param expected_output Expected output vector
     * @return Cost of the network
     */
    double cost(const vector<unsigned char>& input, const vector<double>& expected_output);

    /**
     * @brief Calculate the cost of a dataset
     * @param dataset Dataset
     * @param start_pos Starting position
     * @param batch_size Batch size
     * @return Cost of the dataset
     */
    double cost(const data_set& dataset, int start_pos = 0, int batch_size = 100);

    /**
     * @brief Calculate the gradient of the network (Backpropagation)
     * @param input Input vector
     * @param expected_output Expected output vector
     */
    void calculate_gradient(const vector<unsigned char>& input, const vector<double>& expected_output);

    /**
     * @brief Update the weights of the network (Backpropagation)
     * @param batch_size Size of the batch
     * @param learning_rate Learning rate
     */
    void update_weights(int batch_size, double learning_rate);

    /**
     * @brief Learn from a dataset
     * @param dataset Dataset
     * @param batch_size Size of the batch
     * @param learning_rate Learning rate
     * @param epochs Number of epochs
     */
    void learn(const data_set& dataset, int batch_size = 100, double learning_rate = 0.5, int epochs = 1);

    /**
     * @brief Copy operator
     * @param other Other neural network
     * @return Copied neural network
     */
    n_network& operator=(const n_network& other);
};


#endif
