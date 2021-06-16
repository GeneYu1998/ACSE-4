#pragma once

#include "CUnit.h"

void mark_units(int unit_num);
/**
 * @brief CCircuit
 * A class storing various information, from the initial economic condition, number of units 
 * to the mass flows between them as well as the monetary_value calculated.
 */
class CCircuit {
public:
    static int num_units;// = 10;
    static int cells;
    CUnit *units = NULL;
    Flow Feed, Conc, Tails, Conc_old, Tails_old;
    static double feed_mass_gormanium;// = 10.0;
    static double feed_mass_waste;// = 100.0;
    static double c_rate_gormanium;// = 0.2;
    static double c_rate_waste;// = 0.05;
    static double t_rate_gormanium;// = 0.8;
    static double t_rate_waste;// = 0.95;
    static double value_gormanium;// = 100;
    static double value_waste;// = -500;
    static double worst_case;// = -
    int source;
    double monetary_value;
 
    bool find_conc, find_tails;

    int *circuit_vector = nullptr;

    static void setup_initial_parameters(const int num_units, const double feed_mass_gormanium, const double feed_mass_waste,
                             const double c_rate_gormanium, const double c_rate_waste, const double t_rate_gormanium,
                             const double t_rate_waste, const double value_gormanium,
                             const double value_waste);

    static void setup_CUnits();
    
    CCircuit(){};
    CCircuit(const CCircuit& to_copy)
    {

        this->initialize_units();
        this->initialize_vector();
        this->set_units(to_copy.circuit_vector);
        this->get_circuit_vector();
//        this->monetary_value=monetary_direct_solver(this->circuit_vector, this->num_units * 2 + 1);
    }

    ~CCircuit() {
        if (units != nullptr) {
            delete[] units;
        }
    };

    void initialize_units();
    void set_initial_flow(int unit);
    void clear();
    void save_flow();
    void clear_marks();
    void calculate_monetary_value(bool& diverge, double tolerance = 1e-12, int max_iter = 500);
    void set_units(int *vec);

    void initialize_vector();
    void get_circuit_vector();
    bool Check_Validity();
    void mark_units(int unit_num);
    void generate_random();
    void calculate_monetary_value_directly();
    int* generate_vector();
    static double monetary_direct_solver(int *circuit_vector, int len);
    static void Guassian_elimination(double** a, double* b, int n);

    static void matmul(double** a, double* b, int n);
};
