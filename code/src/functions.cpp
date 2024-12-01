#include <cmath>
#include "functions.h"


double ReLu(double input){
    return input >= 0 ? input : input * 0.01;
}
double d_ReLu(double input){
    return input >= 0 ? 1 : 0;
}


double sig(double input){
    double aux = 1/(1+exp(-input));
    if(aux > MAX_SIG) return MAX_SIG;
    else if(aux < MIN_SIG) return MIN_SIG;
    return aux;
}
double d_sig(double output){
    return output * (1 - output);
}


// int reverseInt (int i) {
//     unsigned char c1, c2, c3, c4;
//     c1 = i & 255, c2 = (i >> 8) & 255, c3 = (i >> 16) & 255, c4 = (i >> 24) & 255;
//     return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
// }
int reverseInt(int i) {
    unsigned char c1, c2, c3, c4;
    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;
    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}