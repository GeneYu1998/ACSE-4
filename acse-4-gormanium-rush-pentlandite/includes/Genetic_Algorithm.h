#pragma once
/**
 * @brief Genetic algorithm
 * Optimise a system represented by a specification vector
 */
class GeneticAlgorithm {

public:
    static void setup_hyper_parameters(int parent_count, double crossover_rate, double mutation_rate);
    void initialize_list(int *vec);

    int select_parents(CCircuit* parents, int n);

    void find_max_index(CCircuit* parents, int& index, double& value);

    void crossover(CCircuit* mother, CCircuit* father);

    void mutate(CCircuit* parent);

    double main_process();
    
    void print_parents();
    void print_next_generation();

    GeneticAlgorithm();
    GeneticAlgorithm(int iter_count, int threshold);
    ~GeneticAlgorithm();

    
    ///collection of valid circuits
    CCircuit* parents = nullptr;
    ///collection of valid circuits in next generation
    CCircuit* next_generation =nullptr;


    /// a set number of iterations
    int iter_count;

    /// the minimum performance gap between two generation of best vector
    double threshold;

    /// the number of offspring/parent n that are evaluated in each generation
    static int parent_count;

    /// the probability of crossing selected parents rather than passing
    /// them into the mutation step unchanged
    static double crossover_rate;

    /// the rate at which mutations are introduced
    static double mutation_rate;

private:


};
