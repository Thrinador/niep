# Experimental NIEP tools 

## Overview of the problems

Collection of tools and software related to solving the nonnegative inverse eigenvalue problem (NIEP) and two major sub problems the symmetric nonnegative inverse eigenvalue problem (SNIEP) and stochastic symmetric nonnegative inverse eigenvalue problem (SSNIEP). 

It is well known that solving the stochastic NIEP is equivalent to solving the NIEP, so all optimizers will use this as the default to save n variables. It is also assumed the the SNIEP is stochastic by default. For the general SNIEP we will use the notation of sub_sniep since to get the optimizer to handle it we use sub-stochastic matrices instead. This gives us the current supported options of 
- niep: Stochastic nonnegative matrices
- sniep: Stochastic symmetric nonnegative matrices
- sub_sniep: Sub-stochastic symmetric nonnegative matrices, scaled so that the perron root is 1. 

## Examples 



# Scripts

## Optimizer

The optimizer attempts to solve the boundary of the coefficients of the characteristic polynomial. It does this by first using the trace as a constraint for finding the min and max of the sum of the 2x2 principal minors (E_2). Then it build a mesh where the trace and E_2 are constraints and it solves for min/max of E_3. It continues this process until all specified E_k's are solved for. This process gets a numerical boundary of the coefficient space, the input points for those are then the matrices that realize them. We can use those matrices to get their spectra, which forms the boundary of the spectra space. 

The file config.toml handles all the specifications on which matrix type to work with, the size of that matrix, tolerances to solve for, and more. 

Once the data is generated, it is all saved in a JSON file in its appropriate matrix type folder. The format is as an array of the following block
'''
{
    "type": "string for min or max",
    "coefficients": [double array],
    "matrix": [double array],
    "eigenvalues": [double array]
},
'''

Note that the matrix is not saved in its entirity, but instead only the free variables are saved. For example in the sniep case only the upper triangle is saved. 

After the JSON is saved the last portion of the optimizer creates plots. These plots will be based on the size of the arrays being worked with, but they will be scatter plots up to 4d (where the 4th dimension is color). The plots will be saved in the same folder of matrix type under the plots folder.

## Extreme points

The other main script is the extreme points finder. For this script you give a collection of extreme points and a data set. It will then say if any of the extreme points can be formed as a convex combination of the others (so that they can be removed). Next it will make a json file of all the points from the provided data that are outside the convex hull. Next, it will then print a configurable number of furthest points from that convex hull. Finally, it will print the exposed faces of the current points.

This script is useful for trying to build the set of extreme points You can start with your guessed points, then add on till you cover all the data. 