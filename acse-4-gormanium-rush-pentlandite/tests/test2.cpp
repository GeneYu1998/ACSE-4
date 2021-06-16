#include <cmath>
#include <iostream>

#include "CUnit.h"
#include "CCircuit.h"
#include "Genetic_Algorithm.h"


void test_main_process() {
    CCircuit::setup_initial_parameters(5, 10.0, 100.0, 0.2, 0.05, 0.8, 0.95, 100, -500);
    CCircuit::setup_CUnits();
    //the number of parent, crossover rate, mutation rate
    GeneticAlgorithm::setup_hyper_parameters(100, 0.8, 0.008);
    //max iteration time, max best vector repetition time
    GeneticAlgorithm test(500, 200);
    //run genetic algorithm
    double res = 0.0;
    for (int i = 0; i < 5; i++) {
        double temp = test.main_process();
        if (temp > res) {
            res = temp;
        }
    }

    std::cout << "Genetic_Algorithm(10) close to 24.8162:\n";
    if (std::fabs(res - 24.8162) < 1.0e-3)
        std::cout << "pass\n";
    else
        std::cout << "fail\n";


    CCircuit::setup_initial_parameters(10, 10.0, 100.0, 0.2, 0.05, 0.8, 0.95, 100, -500);
    res = 0.0;
    for (int i = 0; i < 5; i++) {
        double temp = test.main_process();
        if (temp > res) {
            res = temp;
        }
    }
    std::cout << "Genetic_Algorithm(10) close to 165.753:\n";
    if (std::fabs(res - 165.753) < 1.0e-3)
        std::cout << "pass\n";
    else
        std::cout << "fail\n";

}

int main(int argc, char *argv[]) {
    CCircuit a;
    CCircuit::setup_initial_parameters(10, 10.0, 100.0, 0.2, 0.05, 0.8, 0.95, 100, -500);
    CCircuit::setup_CUnits();


    int vec1[21] = {3, 8, 2, 2, 11, 8, 3, 6, 5, 0,
                    1, 6, 7, 8, 0, 6, 9, 10, 6, 6, 4};

    int vec2[41] = {18, 14, 2, 14, 16, 14, 9, 8,
                    19, 20, 15, 8, 18, 18, 12, 14, 0, 4, 13, 14, 10,
                    3, 11, 19, 17, 18, 21, 4, 14, 8, 3, 20, 8, 14, 7, 5, 6, 14, 1, 8, 5};

    std::cout << "Evaluate_Monetary_value(vec1) close to 165.753:\n";
    if (std::fabs(a.monetary_direct_solver(vec1, 21) - 165.753) < 1.0e-3)
        std::cout << "pass\n";
    else
        std::cout << "fail\n";


    std::cout << "Evaluate_Monetary_value(vec2) close to 680:\n";
    if (std::fabs(a.monetary_direct_solver(vec2, 41) - 680 < 1.0))
        std::cout << "pass\n";
    else
        std::cout << "fail";
}
