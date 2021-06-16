# pragma once

#include<math.h>

/**
 * @brief Flow
 * Representing the mass flow between units.
 */
class Flow {
public:

    double Gormanium = 0;
    double waste = 0;
     
    // setting up flows by input values
    void set(double G, double W) {
        Gormanium = G;
        waste = W;
    }
    
    // setting up flows by a hard copy
    void set(Flow flow) {
        Gormanium = flow.Gormanium;
        waste = flow.waste;
    }
    
    // perform vector addtion for flows
    void plus(Flow feed) {
        Gormanium += feed.Gormanium;
        waste += feed.waste;
    }
   
    // clear flows
    void clear() {
        Gormanium = 0;
        waste = 0;
    }
    
    // evaluating the D_infinity distance for two flows
    double distance(Flow flow1) {
        double G_diff = fabs(flow1.Gormanium - Gormanium);
        double W_diff = fabs(flow1.waste - waste);

        return fmax(G_diff, W_diff);
    }
};

/**
 * @brief CUnit
 * The class for a certain unit in the Circuit
 */
class CUnit {
public:
    //index of the unit to which this unit’s concentrate stream is connected
    int conc_num;
    //index of the unit to which this unit’s tailing stream is connected
    int tails_num;
    //A Boolean that is changed to true if the unit has been seen
    bool mark;

    static double c_rate_gormanium;
    static double c_rate_waste;
    static double t_rate_gormanium;
    static double t_rate_waste;
    Flow feed, feed_old, conc, tails;

    void calculate_outflow();

    void clearAll();

    bool connect_to_conc;
    bool connect_to_tails;
    bool connect_from_source;
};

