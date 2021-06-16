#include "../includes/CUnit.h"

/**
 * Calculate the mass of Gormanium and waste for the 
 * concentration and tails outflows by a given feed.
 * @return void
*/
void CUnit::calculate_outflow()
{
     conc.Gormanium = c_rate_gormanium * feed.Gormanium;
     conc.waste = c_rate_waste * feed.waste;
     tails.Gormanium = t_rate_gormanium * feed.Gormanium;
     tails.waste = t_rate_waste * feed.waste;
}

/**
 * clear all flows stored in a certain unit.
 * @return void
*/
void CUnit::clearAll()
{
     feed.clear();
     feed_old.clear();
     conc.clear();
     tails.clear();
     mark = false;
}
