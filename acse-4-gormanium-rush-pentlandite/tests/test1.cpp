#include <iostream>

#include "../includes/CCircuit.h"
#include "../includes/CUnit.h"
// #include "../src/CCircuit.cpp"
// #include "../src/CUnit.cpp"

int main(int argc, char * argv[]){

    int invalid_1[21] = {0, 10, 8, 8, 6, 10, 5, 7, 10, 2, 5, 11, 7, 0, 11, 10, 0, 2, 7, 8, 7};
    int valid_1[21] = {7, 6, 8, 9, 0, 1, 3, 0, 7, 2, 7, 11, 6, 9, 1, 3, 4, 6, 2, 10, 5};
    int valid[3] = {0,1,2};
    int invalid[3] = {0,2,2};

    CCircuit a;
    a.num_units = 10;
    a.circuit_vector = invalid_1;
	std::cout << "Check_Validity({0, 10, 8, 8, 6, 10, 5, 7, 10, 2, 5, 11, 7, 0, 11, 10, 0, 2, 7, 8, 7}):\n";
    if (a.Check_Validity())
        std::cout << "fail\n";
    else
        std::cout << "pass\n";
    
    a.circuit_vector = valid_1;
	std::cout << "Check_Validity({7, 6, 8, 9, 0, 1, 3, 0, 7, 2, 7, 11, 6, 9, 1, 3, 4, 6, 2, 10, 5}):\n";
    if (a.Check_Validity())
	    std::cout  << "pass\n";
	else
	    std::cout << "fail\n";

        
    //test functionality of validity
    a.num_units = 1;
    a.circuit_vector = valid;
	std::cout << "Check_Validity({0,1,2}):\n";
    if (a.Check_Validity())
	    std::cout  << "pass\n";
	else
	    std::cout << "fail\n";

    a.circuit_vector = invalid;  
    std::cout << "Check_Validity({0,2,2}):\n";
    if (a.Check_Validity())
        std::cout << "fail\n";
    else
        std::cout << "pass\n";
    
    return 0;
}
