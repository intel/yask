#include <iostream>
#include <cstring>
#include "fd_coeff.hpp"
#define MIN(x, y) (((x) < (y)) ? (x): (y))
#define MAX(x, y) (((x) > (y)) ? (x): (y))

using namespace std;

int main()
{

//set the order of the derivative to approximate
//e.g. we want to approximate d^m/dx^m 
const int order = 2;

//set the evaluation point e.g. we want to approximate some derivative f^(m)[eval_point]
//for most application, this is 0
float eval_point = 0;


//Construct a set of points (-h*radius, -h*(radius-1), .. 0, h,..., h*radius) 
//Could pass any arbitrary array grid_points = {x_0, x_1, ... x_n} 
const int radius = 1;
const int num_points = 2*radius+1;
float h = 1;
float coeff[num_points];
float grid_points[num_points]; 

for(int i=0; i<num_points; i++){
grid_points[i] = (i-(num_points-1)/2)*h;
cout << grid_points[i]<<endl;
}


fd_coeff(coeff, eval_point, order, grid_points, num_points);


//TODO: this is hideous, use printf
cout << "The " << order << "^th" << "derivative of f("<< eval_point << ") is: "<<endl;

cout << "f^(" << order << ")@ " << eval_point << "~~ " << coeff[0] << "*f[" << grid_points[0] << "]";

for(int i=1; i<num_points-1; i++){
cout << "+ " << coeff[i] << "*f[" << grid_points[i] << "]";
}

cout << "+" << coeff[num_points-1] << "*f[" << grid_points[num_points-1] << "]" << endl;

return 0;
}
