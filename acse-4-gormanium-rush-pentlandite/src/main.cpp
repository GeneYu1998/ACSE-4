#include <stdio.h>
#include <iostream>

//#include "CCircuit.cpp"
//#include "CUnit.cpp"
//#include "Genetic_Algorithm.cpp"
#include "../includes/CUnit.h"
#include "../includes/CCircuit.h"
#include "../includes/Genetic_Algorithm.h"

using namespace std;

int main(int argc, char* argv[]) {

	srand(time(NULL));
	const int num_unit = 10; ///the number of unit

	CCircuit::setup_initial_parameters(num_unit, 10.0, 100.0, 0.2, 0.05, 0.8, 0.95, 100, -500);
	CCircuit::setup_CUnits();

	//the number of parent, crossover rate, mutation rate
	GeneticAlgorithm::setup_hyper_parameters(120, 0.8, 0.01); //GA parameters after elaborative test, feel free to change
	//max iteration time, max best vector repetition time
	GeneticAlgorithm test(num_unit * 50, 300);
	//run genetic algorithm
	test.main_process();

	return 0;
}
