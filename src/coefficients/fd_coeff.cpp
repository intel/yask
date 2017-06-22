#include <iostream>
#include <cstring>
#include "fd_coeff.hpp"
#define MIN(x, y) (((x) < (y)) ? (x): (y))
#define MAX(x, y) (((x) > (y)) ? (x): (y))

using namespace std;
/*input:coeff=empty coefficient array (or one that can be overwritten)
	eval_point=point at which the derivative is approximated
	order=order of the derivative to approximate (e.g. f'' corresponds to order = 2)
	points=array of points from which to construct the approximation of the derivative. usually an equi-spaced array with points [-radius*h, -(radius-1)*h,...0, ... radius*h] 
	num_points=number of elements in points[], e.g. the number of points used to approximate the derivative. 
	Note: if num_points < order+1, then the coefficients will all be 0

  
  output:void, fills the coefficient array such that 
f^(m)[eval_point] ~~ sum of coeff[i]*f[point[i]] from i = 0 to num_points-1
*/
	
void fd_coeff(float coeff[], float eval_point, int order, float points[], int num_points)
{



float c1, c2, c3;
float x_0=eval_point;
float center=0;




//need a vector which stores fd_coefficients
//coefficients are in d[end][end][i]
float d[order+1][num_points][num_points];
memset(d, 0, sizeof(d));
d[0][0][0]=1;
c1 = 1.f;


for(int n=1; n<=num_points-1;n++){
c2=1.f;
	for(int v=0; v<=n-1; v++){
	c3 = points[n] - points[v];
	c2 = c2*c3;
		for(int m=0; m<=MIN(n, order); ++m){
		d[m][n][v] = (points[n]-x_0)*d[m][n-1][v] - m*d[m-1][n-1][v];
		d[m][n][v] *= 1.f/c3;
		}
	}
	for(int m=0; m<= MIN(n, order); m++){
	d[m][n][n] = m*d[m-1][n-1][n-1] - (points[n-1]-x_0)*d[m][n-1][n-1];
	d[m][n][n] *= c1/c2;
	}
c1=c2;
}

cout << "DONE COMPUTING COEFFICIENTS"<< endl;

for(int i=0; i<num_points; i++){
coeff[i] = d[order][num_points-1][i];

}

}
