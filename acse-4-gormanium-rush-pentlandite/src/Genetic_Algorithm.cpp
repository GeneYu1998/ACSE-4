#include <cmath>
#include <ctime>
#include <iostream>
#include <cstdlib>
#include <iomanip>
#include <cstring>
#include <sstream>
#include <fstream>
//#include <omp.h>
#include "../includes/CUnit.h"
#include "../includes/CCircuit.h"
#include "../includes/Genetic_Algorithm.h"

using namespace std;

//initialize static variable
int GeneticAlgorithm::parent_count;
double GeneticAlgorithm::crossover_rate;
double GeneticAlgorithm::mutation_rate;
//int thread_num = omp_get_max_threads();

/**
*  set up hyper parameters
 * @param parent_count
 * the number of offspring n that are evaluated in each generation;
 * @param crossover_rate
 * the probability of crossing selected parents rather than passing them into the mutation step
 * unchanged (a recommended range is between 0.8 and 1)
 * @param mutation_rate
 * the rate at which mutations are introduced (recommended probabilities of 1% or lower).
 * @return void
*/
void GeneticAlgorithm::setup_hyper_parameters(int parent_count, double crossover_rate, double mutation_rate) {
    GeneticAlgorithm::parent_count = parent_count;
    GeneticAlgorithm::crossover_rate = crossover_rate;
    GeneticAlgorithm::mutation_rate = mutation_rate;
}

/**
 * Generate one random CCircuit vector.
 * @param vec
 */
void GeneticAlgorithm::initialize_list(int *vec) {

    vec[0] = rand() % CCircuit::num_units; //between 0 and num_units-1
    int rand1;
    int rand2;
    for (int i = 0; i < CCircuit::num_units; i++) {
        while ((rand1 = rand() % (CCircuit::num_units + 2)) == i); //rand1 not equal to i
        while ((rand2 = rand() % (CCircuit::num_units + 2)) == i ||
               rand2 == rand1); //rand2 not equal to i and not equal to rand1
        vec[i * 2 + 1] = rand1;
        vec[i * 2 + 2] = rand2;
    }
}

/**
 * Use tournament selection select parents
 * @param parents collection of valid circuits
 * @param n select n from parent then use largest one as parent
 * @return parent
 */
int GeneticAlgorithm::select_parents(CCircuit *parents, int n) {
    int res = -1;
    for (int i = 1; i <= n; i++) {
        if (res == -1) {
            res = rand() % parent_count;
        } else {
            int cur = rand() % parent_count;
            if (parents[cur].monetary_value > parents[res].monetary_value) {
                res = cur;
            }
        }
    }
    return res;
}

/**
 * Find the best vector.
 * @param parents collection of valid circuits
 * @param index index of best vector
 * @param value value of best vector
 * @return void
 */
void GeneticAlgorithm::find_max_index(CCircuit *parents, int &index, double &value) {
    value = parents[0].monetary_value;
    index = 0;
    for (int i = 1; i < parent_count; i++)
        if (parents[i].monetary_value > value) {
            value = parents[i].monetary_value;
            index = i;
        }
}

/**
 * @mother and @father are selected to cross, a random point in the vector is chosen and all of
 * the values before that point are swapped with the corresponding points in the other vector.
 * @param mother selected parent
 * @param father selected parent
 * @return void
 */
void GeneticAlgorithm::crossover(CCircuit *mother, CCircuit *father) {
    if ((rand() % 1000) < int(1000 * this->crossover_rate)) {
        int change_site;
        int temp;

        change_site = rand() % (2 * CCircuit::num_units) + 1;
        //cout << "change site is " << change_site << endl;
        for (int i = 0; i < change_site; ++i) {
            temp = mother->circuit_vector[i];
            mother->circuit_vector[i] = father->circuit_vector[i];
            father->circuit_vector[i] = temp;
        }

        mother->set_units(mother->circuit_vector);
        father->set_units(father->circuit_vector);
    }

}

/**
 * Go over each of the numbers in the vector and decide whether to mutate them
 * @param parent
 * @return void
 */
void GeneticAlgorithm::mutate(CCircuit *parent) {
    for (int i = 0; i < 2 * CCircuit::num_units + 1; ++i) {
        if ((rand() % 1000) < int(1000 * mutation_rate)) {
            parent->circuit_vector[i] = rand() % CCircuit::num_units;
        }
    }
    parent->set_units(parent->circuit_vector);
}

/**
 * The main process of the basic genetic algorithm
 * @return void
 */
double GeneticAlgorithm::main_process() {

    srand((unsigned) time(NULL));
    // Step1. Start with the vectors representing the initial random collection of valid circuits.
    int *vector = new int[2 * CCircuit::num_units + 1];

//pragma omp parallel for num_threads(thread_num)
    for (int i = 0; i < parent_count; i++) {
        parents[i].initialize_vector();
        parents[i].initialize_units();
        do {
            initialize_list(vector);
            parents[i].set_units(vector);
            parents[i].get_circuit_vector();
            //parents[i].monetary_value = CCircuit::monetary_direct_solver(parents[i].circuit_vector, 2 * CCircuit::num_units + 1);
            parents[i].calculate_monetary_value_directly();
        } while (!parents[i].Check_Validity());
    }
    
    stringstream fname;
	fstream f1;
    int iter = 0;
    double best_value;
    double last_best_value;
    int count_best_value = 0;
    f1.open("./output.txt", ios_base::out);
    f1.close();
    f1.open("./output.txt", ios::app);
    while (iter++ <= iter_count && count_best_value < threshold) {
        std::cout << endl << "Generation: " << iter << endl;

        // Step2. Calculate the fitness value for each of these vectors.      
        for (int i = 0; i < parent_count; i++) {
            parents[i].monetary_value = CCircuit::monetary_direct_solver(parents[i].circuit_vector,
                                                                         parents[i].num_units * 2 + 1);

        }

        // Step3. Take the best vector (the one with the highest fitness value) into the child list unchanged       
        int index;
        find_max_index(parents, index, best_value);
        int nextIndex = 0;
        CCircuit cCircuit(parents[index]);
        next_generation[nextIndex++] = cCircuit;

        std::cout << "The best monetary value in this generation is: " << best_value << endl;
        std::cout << "The Vector is: ";
        for (int i = 0; i < 2 * CCircuit::num_units + 1; i++) {
            cout << setw(4) << cCircuit.circuit_vector[i];
        }

	    for (int i = 0; i < 2 * CCircuit::num_units + 1; i++) f1 << cCircuit.circuit_vector[i] << " ";
        f1 << best_value << endl;


        if (count_best_value == 0) {
            last_best_value = best_value;
            count_best_value++;
        } else if (fabs(last_best_value - best_value) < 1e-6)
            count_best_value++;
        else
            count_best_value = 0;
        cout << endl;

        // Repeat this process from step 4 until there are n child vectors
        

//#pragma omp parallel num_threads(thread_num)
        //{
            while (nextIndex < parent_count) {
                // Step4. Select a pair of the parent vectors with a
                // probability that depends on the fitness value.

                int mom_index = select_parents(parents, 2);
                int dad_index;

                do { dad_index = select_parents(parents, 2); } while (dad_index == mom_index);
                CCircuit momCircuit(parents[mom_index]);
                CCircuit dadCircuit(parents[dad_index]);

                // Step5.Randomly decide if the parents should crossover.
                if ((rand() % 1000) < int(1000 * crossover_rate)) {
                    crossover(&momCircuit, &dadCircuit);
                }
                // Step6.Go over each of the numbers in both the vectors and decide whether to mutate them
                mutate(&momCircuit);
                mutate(&dadCircuit);

                // Step7.Check that each of these potential new vectors are valid and,
                // if they are, add them to the list of child vectors.
                if (nextIndex < parent_count && momCircuit.Check_Validity()) {
//#pragma omp single nowait
                    //{
                        //next_generation[nextIndex++] = CCircuit(momCircuit);
                        next_generation[nextIndex++] = momCircuit;
                    //}
                //}
                if (nextIndex < parent_count && dadCircuit.Check_Validity()) {
//#pragma omp single nowait
                    //{
                        //next_generation[nextIndex++] = CCircuit(dadCircuit);
                        next_generation[nextIndex++] = dadCircuit;
                    //}
                }
            }
        }
        CCircuit *temp = parents;
        parents = next_generation;
        next_generation = temp;        
    }
    
    f1.close();
    std::cout << endl << "Reach Convergence Criteria, Iteration Finished!" << endl;
    delete[] vector;

    return best_value;
}

/**
 * Print value in parents
 * @return void
 */
void GeneticAlgorithm::print_parents() {

    for (int j = 0; j < parent_count; j++) {
        std::cout << j << ": ";
        for (int i = 0; i < CCircuit::num_units * 2 + 1; ++i) {
            std::cout << setw(2) << parents[j].circuit_vector[i] << " ";
        }
        std::cout << setw(10) << "value: " << parents[j].monetary_value;
        std::cout << endl;
    }
}

/**
 * Print value in next generation
 * @return void
 */
void GeneticAlgorithm::print_next_generation() {

    for (int j = 0; j < parent_count; j++) {
        std::cout << j << ": ";
        for (int i = 0; i < CCircuit::num_units * 2 + 1; ++i) {
            std::cout << setw(2) << next_generation[j].circuit_vector[i] << " ";
        }
        std::cout << setw(10) << "value: " << next_generation[j].monetary_value;
        std::cout << endl;
    }
}

/**
 * constructor
 */
GeneticAlgorithm::GeneticAlgorithm() {
    parents = new CCircuit[parent_count];
    next_generation = new CCircuit[parent_count];
    iter_count = 100; //max iteration
}

/**
 * constructor
 * @param iter_count max iteration
 * @param threshold the minimum performance gap between two generation of best vector
 */
GeneticAlgorithm::GeneticAlgorithm(int iter_count, int threshold) : GeneticAlgorithm() {
    this->iter_count = iter_count; //max iteration
    this->threshold = threshold;

}

/**
 * destructor
 */
GeneticAlgorithm::~GeneticAlgorithm() {

}




