#ifndef FUNCTIONS_H
#define FUNCTIONS_H

// This shoyld be in its own namespace, but i was stupid back then

/**
 * @brief ReLu activation function
 * @param input the input value
 */
double ReLu(double input);

/**
 * @brief Derivative of ReLu activation function
 * @param input the input value
 */
double d_ReLu(double input);

//Constants outside a namespace? Jeez...
const double MAX_SIG = 0.99; //*< Max value of the sigmoid function */
const double MIN_SIG = 0.01; //*< Min value of the sigmoid function */

/**
 * @brief Sigmoid activation function
 * @param input the input value
 */
double sig(double input);

/**
 * @brief Derivative of Sigmoid activation function
 * @param input the input value
 */
double d_sig(double input);


/**
 * @brief Struct that holds the activation function and its derivative
 */
struct activation{
    
    double (*function)(double); //*< Activation function */
    double (*derivative)(double); //*< Derivative of the activation function */

    /**
     * @brief Constructor
     * @param function Activation function
     * @param derivative Derivative of the activation function
     */
    explicit activation(double (*function)(double) = ReLu, double (*derivative)(double)  = d_ReLu) {
        this->function = function;
        this->derivative = derivative;
    }

    /**
     * @brief Copy constructor
     * @param other Other activation struct
     */
    activation& operator=(const activation& other)= default;
};

//Default activation functions
static const activation ReLu_activation(ReLu, d_ReLu); //*< ReLu activation function */
static const activation sig_activation(sig, d_sig); //*< Sigmoid activation function */

/**
 * @brief Transforms an integer from big endian to little endian
 */
int reverseInt (int i);

#endif //STANDARNEURONLNETWORK_FUNCTIONS_H
