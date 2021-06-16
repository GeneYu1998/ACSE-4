#include <vector>
#include <stdio.h>
#include <math.h>
#include <iostream>
#include "../includes/CCircuit.h"

using namespace std;

typedef struct node {

	bool c_val = false;
	bool t_val = false;
	int unit;
	int p_num = 0;
	int* parent;

} node;

//initialize static variable
int CCircuit::cells;
int CCircuit::num_units;
double CCircuit::feed_mass_gormanium;
double CCircuit::feed_mass_waste;
double CCircuit::c_rate_gormanium;
double CCircuit::c_rate_waste;
double CCircuit::t_rate_gormanium;
double CCircuit::t_rate_waste;
double CCircuit::value_gormanium;
double CCircuit::value_waste;
double CCircuit::worst_case;
double CUnit::c_rate_gormanium;
double CUnit::c_rate_waste;
double CUnit::t_rate_gormanium;
double CUnit::t_rate_waste;

/**
*  set initial parameters
 * @param num_units
 * 
 * @param feed_mass_gormanium
 * 
 * @param feed_mass_waste
 * 
 * @param c_rate_gormanium
 * 
 * @param c_rate_waste
 * 
 * @param t_rate_gormanium
 * 
 * @param t_rate_waste
 * 
 * @param value_gormanium
 * 
 * @param value_waste
 * 
 * @return void
*/
void CCircuit::setup_initial_parameters(const int num_units, double feed_mass_gormanium, double feed_mass_waste,
	double c_rate_gormanium, double c_rate_waste, double t_rate_gormanium,
	double t_rate_waste, double value_gormanium,
	double value_waste) {
	CCircuit::num_units = num_units;
	CCircuit::feed_mass_gormanium = feed_mass_gormanium;
	CCircuit::feed_mass_waste = feed_mass_waste;
	CCircuit::c_rate_gormanium = c_rate_gormanium;
	CCircuit::c_rate_waste = c_rate_waste;
	CCircuit::t_rate_gormanium = t_rate_gormanium;
	CCircuit::t_rate_waste = t_rate_waste;
	CCircuit::value_gormanium = value_gormanium;
	CCircuit::value_waste = value_waste;
	CCircuit::worst_case = value_waste * feed_mass_waste;
	CCircuit::cells = CCircuit::num_units * 2 + 1;
}

/**
*  set CUnit parameters according to CCircuit
 * @return void
*/
void CCircuit::setup_CUnits() {
	CUnit::c_rate_gormanium = CCircuit::c_rate_gormanium;
	CUnit::c_rate_waste = CCircuit::c_rate_waste;
	CUnit::t_rate_gormanium = CCircuit::t_rate_gormanium;
	CUnit::t_rate_waste = CCircuit::t_rate_waste;
}

/**
*  initialize unit according to unit number
 * @return void
*/
void CCircuit::initialize_units() {
	units = new CUnit[num_units];
}

/**
*  initialize vector
 * @return void
*/
void CCircuit::initialize_vector() {
	if (circuit_vector != nullptr) { delete[] circuit_vector; }

	this->circuit_vector = new int[2 * num_units + 1];
}

/**
*  clear unit, concentration, tailing
 * @return void
*/
void CCircuit::clear() {
	for (int i = 0; i < num_units; i++)
		units[i].clearAll();

	Conc.clear();
	Tails.clear();
}

void CCircuit::save_flow() {
	for (int i = 0; i < num_units; i++) {
		units[i].feed_old = units[i].feed;
		units[i].feed.clear();
	}

	Conc_old = Conc;
	Conc.clear();
	Tails_old = Tails;
	Tails.clear();
}


/**
*  Calculate monetary value, using iteration method
 * @param diverge
 * boolean indicating whether flow in the circuit is diverging or not.
 * @param tolerance
 * which is setting for the stopping criteria, 
 * if the change over time step is small,
 * we achieve our balance state.
 * @param max_iter
 * we do not want our method taking two much time,
 * if some case converge too slow and met max iteraton times we set,
 * the method would quit.
 * @return void
*/
void CCircuit::calculate_monetary_value(bool& diverge, double tolerance, int max_iter)
{
	double res = 1;
	int iter = 0;
	int conc_num;
	int tails_num;
	int current_max;
	diverge = false;

	// set the initial feed state
	clear();
	Feed.set(CCircuit::feed_mass_gormanium, CCircuit::feed_mass_waste);
	units[source].feed.set(Feed);

	// do the flow iteration
	while (res >= tolerance && iter < max_iter)
	{
		// produce conc and tails for each units[i]
		for (int i = 0; i < num_units; ++i)
		{
			units[i].calculate_outflow();
		}

		save_flow();

		// remember the Feed of source is always constant!!
		units[source].feed.set(Feed);
		// cout << units[source].feed.Gormanium << " " << units[source].feed.waste << endl;

		// passing conc and tails to the feed of the next unit
		for (int i = 0; i < num_units; ++i)
		{
			conc_num = units[i].conc_num;

			if (conc_num < num_units)
			{
				units[conc_num].feed.plus(units[i].conc);
			}
			else if (conc_num == num_units) Conc.plus(units[i].conc);
			else Tails.plus(units[i].conc);
			tails_num = units[i].tails_num;
			if (tails_num < num_units)
			{
				units[tails_num].feed.plus(units[i].tails);
			}
			if (tails_num == num_units) Conc.plus(units[i].tails);
			else Tails.plus(units[i].tails);
		}

		res = 0;

		// adding the changes of the feeds of the units. (d_infinity - norm)
		for (int i = 0; i < num_units; i++)
		{
			res = units[i].feed.distance(units[i].feed_old);
        // checking if accumulation happens?
        if((current_max = 1000 * fmax(units[i].feed_old.Gormanium, 
                units[i].feed_old.waste))!= 0 && current_max < res)
        { 
                diverge = true;
                iter = max_iter;
                monetary_value = CCircuit::worst_case;
                break;
        }
		}

		// adding the changes of the Conc and Tails. (d_infinity - norm)
		res = fmax(res, Conc.distance(Conc_old));
		res = fmax(res, Tails.distance(Tails_old));

		iter++;
		//cout << "iteration: " << iter << " ";
		//cout << "Conc.G " << Conc.Gormanium << "  Conc.W " << Conc.waste << endl;
	}

	if (iter < max_iter)
	{
		monetary_value = CCircuit::value_gormanium * Conc.Gormanium
			+ CCircuit::value_waste * Conc.waste;
	}
	else
	{
		diverge = true;
		monetary_value = CCircuit::worst_case;
	}
}


/**
*  set up units for a CCircuit class according to vector value
 * @param vec
 * the input vector, indicating a configaration. 
 * @return void
*/
void CCircuit::set_units(int* vec) {
	clear();

	source = vec[0];

	for (int i = 0; i < num_units; i++) {
		units[i].conc_num = vec[i * 2 + 1];
		units[i].tails_num = vec[i * 2 + 2];
	}
}

/**
*  Extract the configaration vector from a CCircuit class.
 *
 * @return void
*/
void CCircuit::get_circuit_vector() {

	circuit_vector[0] = source;

	for (int i = 0; i < num_units; i++) {
		circuit_vector[i * 2 + 1] = units[i].conc_num;
		circuit_vector[i * 2 + 2] = units[i].tails_num;
	}
}

/**
 * @brief this function is to check whether the units connect to the tailing
 *
 * @param tail tail number to locate the tail node
 * @param nodes address of nodes array
 * @return stop while backs to the tail
 * @note this function backs recursively check the parents units, start from
 * tail, and mark parent units true which indicates that unit has connected to tail
 */
void check_tailing(int tail, node* nodes) {
	if (!nodes[tail].t_val || tail == CCircuit::num_units) {
		nodes[tail].t_val = true;
		for (int i = 0; i < nodes[tail].p_num; i++) {
			check_tailing(nodes[tail].parent[i], nodes);
		}
	}
	else if (nodes[tail].t_val) {
		return;
	}
}

/**
 * @brief this function is to check whether the units connect to the tailing
 *
 * @param concentrate concentrate number to locate the concentrate node
 * @param nodes address of nodes array
 * @return stop while backs to the concentrate
 * @note this function backs recursively check the parents units, start from
 * tail, and mark parent units true which indicates that unit has connected to concentrate
 */
void check_concentrate(int con, node* nodes) {
	if (!nodes[con].c_val || con == CCircuit::num_units + 1) {
		nodes[con].c_val = true;
		for (int i = 0; i < nodes[con].p_num; i++) {
			check_concentrate(nodes[con].parent[i], nodes);
		}
	}
	else if (nodes[con].c_val) {
		return;
	}
}

/**
 * @brief this function is to check validility
 * @return true if valid
 * @note this function firsly construct the units array, if the unit is self connected,
 * directly return false, then the function check whether all units are connected to
 * tail and concentrate.
 */
bool CCircuit::Check_Validity() {

	node* nodes = new node[num_units + 2];
	auto* parent = new int* [num_units + 2];

	for (int i = 0; i < num_units + 2; i++) {
		node* newnode = new node();
		newnode->unit = i;
		parent[i] = new int[num_units];
		newnode->parent = parent[i];
		nodes[i] = *newnode;
		delete newnode;
	}

	for (int i = 0; i < num_units; i++) {

		int left = circuit_vector[i * 2 + 1];
		int right = circuit_vector[i * 2 + 2];
		if (left == i || right == i) {
			for (int i = 0; i < num_units + 2; i++)
			{
				delete[] parent[i];
			}
			delete[] parent;
			delete[] nodes;
			return false;
		}
		nodes[left].parent[nodes[left].p_num] = i;
		nodes[left].p_num++;
		nodes[right].parent[nodes[right].p_num] = i;
		nodes[right].p_num++;
	}
	check_tailing(num_units, nodes);
	check_concentrate(num_units + 1, nodes);

	for (int i = 0; i < num_units; i++) {
		if (!nodes[i].c_val || !nodes[i].t_val) {
			for (int i = 0; i < num_units + 2; i++)
			{
				delete[] parent[i];
			}
			delete[] parent;
			delete[] nodes;
			return false;
		}
		if (i != circuit_vector[0] && nodes[i].p_num == 0) {
			for (int i = 0; i < num_units + 2; i++)
			{
				delete[] parent[i];
			}
			delete[] parent;
			delete[] nodes;
			return false;
		}
	}
	for (int i = 0; i < num_units + 2; i++)
	{
		delete[] parent[i];
	}
	delete[] parent;
	delete[] nodes;
	return true;
}

/**
 * Do Gaussian elimination, 
 * for solving the linear equation Ax = b
 * @param a
 * the matrix A on the left hand side 
 * @param b
 * the RHS vector 
 * @param n the size of the problem
 * @return void
*/
void CCircuit::Guassian_elimination(double** a, double* b, int n)
{
	int i, j, k;
	double temp;

	double* c = new double[n];

	for (k = 0; k < n; k++)
	{
		//evaluate divisor
		for (i = k + 1; i < n; i++)
			c[i] = a[i][k] / a[k][k];

		//row operation(elimination)
		for (i = k + 1; i < n; i++)
		{
			for (j = 0; j < n; j++)
			{
				a[i][j] -= c[i] * a[k][j];
			}
			b[i] -= c[i] * b[k];
		}
	}

	//evaluate last element
	b[n - 1] /= a[n - 1][n - 1];

	//back substitution
	for (i = n - 2; i >= 0; i--)
	{
		temp = 0;
		for (j = i + 1; j < n; j++)
		{
			temp += a[i][j] * b[j];
		}
		b[i] -= temp;
		b[i] /= a[i][i];
	}
	delete[] c;
}

/**
 * performing left Matrix-vector multiplication
 * @param a
 * the matrix
 * @param b
 * the vector to be mutiplicated
 * @param n
 * the size of the problem
 * @return void
*/
void CCircuit::matmul(double** a, double* b, int n)
{
	auto x = new double[n];

	for (int i = 0; i < n; ++i) x[i] = 0;

	for (int i = 0; i < n; ++i)
	{
		for (int j = 0; j < n; ++j)
			x[i] += a[i][j] * b[j];
	}

	for (int i = 0; i < n; ++i)
	{
		b[i] = x[i];
		if (fabs(b[i]) < 1e-3) b[i] = 0;
	}

	delete[] x;
}

/**
 * generate a random configuration for a CCircuit object, 
 * which is likely to be valid  
 * @return void
*/
void CCircuit::generate_random()
{
	source = 0; // rand() % num_units; (turn-on to enable random source number)

	// at least one unit to concentrate
	int conc_unit = rand() % num_units;
	units[conc_unit].conc_num = num_units;
	//while((units[conc_unit].tails_num = rand() % (num_units + 2)) == num_units || units[conc_unit].tails_num == conc_unit);

	// at least one unit to tailing
	int tail_unit = rand() % num_units; //while((tail_unit = rand() % num_units) == conc_unit);
	units[tail_unit].tails_num = num_units + 1;
	//while((units[tail_unit].conc_num = rand() % (num_units + 2)) == num_units + 1 || units[tail_unit].conc_num == tail_unit);

	for (int i = 0; i < num_units; ++i)
	{
		if (i != conc_unit && i != tail_unit)
		{
			while ((units[i].conc_num = rand() % (num_units + 2)) == i);
			// prevent a CUnit being a "pipe".
			while ((units[i].tails_num = rand() % (num_units + 2)) == units[i].conc_num || units[i].tails_num == i);
		}
	}
}

/**
 * return a copy for the configuration vector for a CCircuit object.
 * @return vector
*/
int* CCircuit::generate_vector()
{
	auto vec = new int[2 * num_units + 1];
	vec[0] = source;

	for (int i = 0; i < num_units; i++)
	{
		vec[i * 2 + 1] = units[i].conc_num;
		vec[i * 2 + 2] = units[i].tails_num;
	}

	return vec;
}

/**
 * The direct linear solver to calculate the monetary value of the concentration
 * for a CCircuit object, overwrting the monetary_value it stored.
 * @return void
*/
void CCircuit::calculate_monetary_value_directly()
{
	int n = CCircuit::cells + 3;

	auto A = new double* [n];
	auto b = new double[n];

	// initialize A and b
	for (int i = 0; i < n; ++i)
	{
		A[i] = new double[n];
		if (i == source * 2) b[i] = -CCircuit::feed_mass_gormanium;
		else if (i == source * 2 + 1) b[i] = -CCircuit::feed_mass_waste;
		else b[i] = 0;
		for (int j = 0; j < n; ++j) A[i][j] = 0;
	}

	// setting up A
	for (int i = 0; i < num_units; ++i)
	{
		A[2 * units[i].conc_num][2 * i] = CCircuit::c_rate_gormanium;
		A[2 * units[i].conc_num + 1][2 * i + 1] = CCircuit::c_rate_waste;
		A[2 * units[i].tails_num][2 * i] = CCircuit::t_rate_gormanium;
		A[2 * units[i].tails_num + 1][2 * i + 1] = CCircuit::t_rate_waste;
	}

	CCircuit::matmul(A, b, n);

	for (int k = 0; k < n; ++k)
		A[k][k] -= 1;

	CCircuit::Guassian_elimination(A, b, n);

	monetary_value = 100 * b[2 * num_units] - 500 * b[2 * num_units + 1];
	for (int i = 0; i < n; i++)
	{
		delete[] A[i];
	}
	delete[] A;
	delete[] b;
}

/**
 * @param circuit_vector
 * The direct linear solver to calculate the monetary value of the concentration
 * for a configuration vector, returning the monetary_value as a double.
 * @param len
 * The length of the Circuit configuration vector
 * @return momentary value
*/
double CCircuit::monetary_direct_solver(int* circuit_vector, int len)
{
	int conc, tails;
	int num_units = int((len - 1) / 2);
	int source_num = circuit_vector[0];
	int n = len + 3;

	auto A = new double* [n];
	auto b = new double[n];

	// initialize A and b
	for (int i = 0; i < n; ++i)
	{
		A[i] = new double[n];
		if (i == source_num * 2) b[i] = -CCircuit::feed_mass_gormanium;
		else if (i == source_num * 2 + 1) b[i] = -CCircuit::feed_mass_waste;
		else b[i] = 0;
		for (int j = 0; j < n; ++j) A[i][j] = 0;
	}

	// setting up A
	for (int i = 0; i < num_units; ++i)
	{
		conc = circuit_vector[2 * i + 1];
		tails = circuit_vector[2 * i + 2];

		A[2 * conc][2 * i] = CCircuit::c_rate_gormanium;
		A[2 * conc + 1][2 * i + 1] = CCircuit::c_rate_waste;
		A[2 * tails][2 * i] = CCircuit::t_rate_gormanium;
		A[2 * tails + 1][2 * i + 1] = CCircuit::t_rate_waste;
	}

	CCircuit::matmul(A, b, n);

	for (int k = 0; k < n; ++k)
		A[k][k] -= 1;

	CCircuit::Guassian_elimination(A, b, n);

	double conc_g = b[2 * num_units];
	double conc_w = b[2 * num_units + 1];
	double momentary_value = 100 * conc_g - 500 * conc_w;
	for (int i = 0; i < n; i++)
	{
		delete[] A[i];
	}
	delete[] A;
	delete[] b;
	return momentary_value;
	//cout << "Conc.G: " << conc_g << "Conc.w: " << conc_w << "momentary_value: " << momentary_value;
}
