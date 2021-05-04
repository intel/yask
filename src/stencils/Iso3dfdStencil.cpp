/*****************************************************************************

YASK: Yet Another Stencil Kit
Copyright (c) 2014-2021, Intel Corporation

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to
deal in the Software without restriction, including without limitation the
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
sell copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

* The above copyright notice and this permission notice shall be included in
  all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
IN THE SOFTWARE.

*****************************************************************************/

// Implement isotropic 3D finite-difference (FD) stencil, nth-order accurate in
// space (where n = 2 * radius) and 2nd-order accurate in time.
// See https://software.intel.com/en-us/articles/eight-optimizations-for-3-dimensional-finite-difference-3dfd-code-with-an-isotropic-iso.

// YASK stencil solution(s) in this file will be integrated into the YASK compiler utility.
#include "yask_compiler_api.hpp"
using namespace std;
using namespace yask;

// Create an anonymous namespace to ensure that types are local.
namespace {

    class Iso3dfdStencil : public yc_solution_with_radius_base {

    protected:

        // Indices & dimensions.
        yc_index_node_ptr t = new_step_index("t");           // step in time dim.
        yc_index_node_ptr x = new_domain_index("x");         // spatial dim.
        yc_index_node_ptr y = new_domain_index("y");         // spatial dim.
        yc_index_node_ptr z = new_domain_index("z");         // spatial dim.

        // Vars.
        yc_var_proxy p =
            yc_var_proxy("p", get_soln(), { t, x, y, z }); // time-varying 3D var.
        yc_var_proxy v =
            yc_var_proxy("v", get_soln(), { x, y, z }); // constant 3D var = (c(x,y,z)^2 * delta_t^2).

    public:

        // For this stencil, the 'radius' is the number of FD coefficients on
        // either side of center in each spatial dimension.  For example,
        // radius=8 implements a 16th-order accurate FD stencil.  
        // The accuracy in time is fixed at 2nd order.
        Iso3dfdStencil(string suffix="", int radius=8) :
            yc_solution_with_radius_base("iso3dfd" + suffix, radius) { }
        virtual ~Iso3dfdStencil() { }

        // Define RHS expression for p at t+1 based on values from v and p at t.
        virtual yc_number_node_ptr get_next_p() {

            // Grid spacing.
            // In this implementation, it's a constant.
            // Could make this a YASK variable to allow setting at run-time.
            double delta_xyz = 50.0;
            double d2 = delta_xyz * delta_xyz;
        
            // Spatial FD coefficients for 2nd derivative.
            vector<double> coeff;
            size_t rad = get_radius();
            if (rad > 0)
                coeff = get_center_fd_coefficients(2, get_radius());
            else
                coeff.push_back(1.0); // Dummy value for zero radius (not FD).

            for (size_t i = 0; i < coeff.size(); i++) {

                // Need 3 copies of center sample for x, y, and z FDs.
                // Center coeff is at index 'rad'.
                if (i == rad)
                    coeff[i] *= 3.0;

                // Divide each by delta_xyz^2.
                coeff[i] /= d2;
            }

            // Calculate FDx + FDy + FDz.
            // Start with center value multiplied by coeff 0.
            auto fd_sum = p(t, x, y, z) * coeff[rad];

            // Add values from x, y, and z axes multiplied by the
            // coeff for the given radius.
            for (int r = 1; r <= get_radius(); r++) {

                // Add values from axes at radius r.
                fd_sum += (
                           // x-axis.
                           p(t, x-r, y, z) +
                           p(t, x+r, y, z) +

                           // y-axis.
                           p(t, x, y-r, z) +
                           p(t, x, y+r, z) +

                           // z-axis.
                           p(t, x, y, z-r) +
                           p(t, x, y, z+r)

                           ) * coeff[rad + r]; // R & L coeffs are identical.
            }

            // Wave equation is:
            // 2nd_time_derivative(p) = c^2 * laplacian(p).
            // See https://en.wikipedia.org/wiki/Wave_equation.
        
            // For this implementation, we are fixing the accuracy-order in time
            // to 2 and using the known FD coefficients (1, -2, 1) to solve the
            // equation manually.  But we could parameterize by accuracy-order
            // in time as well, starting with a call to
            // 'get_forward_fd_coefficients(2, time_order)' for the temporal FD
            // coefficients.

            // So, wave equation with FD approximations is:
            // (p(t+1) - 2 * p(t) + p(t-1)) / delta_t^2 = c^2 * fd_sum.

            // Solve for p(t+1):
            // p(t+1) = 2 * p(t) - p(t-1) + c^2 * fd_sum * delta_t^2.

            // Let v = c^2 * delta_t^2 for each var point.
            auto next_p = (2.0 * p(t, x, y, z)) -
                p(t-1, x, y, z) + (fd_sum * v(x, y, z));

            return next_p;
        }

        // Define equation for p at t+1 based on values from v and p at t.
        virtual void define() override {

            // Get equation for RHS.
            auto next_p = get_next_p();

            // Define the value at t+1 to be equal to next_p.
            // Since this implements the finite-difference method, this
            // is actually an approximation.
            p(t+1, x, y, z) EQUALS next_p;

            // Apply BKCs.
            set_configs();
        }

        // Set BKC (best-known configs) found by automated and/or manual
        // tuning. They are only applied for certain target configs.
        virtual void set_configs() {
            auto soln = get_soln(); // pointer to compile-time soln.

            // BKCs are only for SP FP (4B) and radius 8.
            if (soln->get_element_bytes() == 4 &&
                get_radius() == 8) {

                // Compile-time defaults, e.g., folding and prefetching.
                if (soln->is_target_set()) {
                    auto target = soln->get_target();
                    if (target == "knl") {
                        soln->set_fold_len(x, 2);
                        soln->set_fold_len(y, 8);
                        soln->set_cluster_mult(x, 2);
                        soln->set_prefetch_dist(1, 1);
                        soln->set_prefetch_dist(2, 0);
                    }
                    else if (target == "avx2") {
                        soln->set_prefetch_dist(1, 0);
                        soln->set_prefetch_dist(2, 0);
                    }
                }

                // Kernel run-time defaults, e.g., block-sizes.
                // This code is run immediately after 'kernel_soln' is created.
                soln->CALL_AFTER_NEW_SOLUTION
                    (
                     // Check CPU target at kernel run-time.
                     if (!kernel_soln.is_offloaded()) {
                         auto isa = kernel_soln.get_target();
                         if (isa == "knl") {
                             kernel_soln.set_block_size("x", 160);
                             kernel_soln.set_block_size("y", 256);
                             kernel_soln.set_block_size("z", 96);
                         }
                         else if (isa == "avx2") {
                             kernel_soln.set_block_size("x", 48);
                             kernel_soln.set_block_size("y", 64);
                             kernel_soln.set_block_size("z", 112);
                         }
                         else {
                             kernel_soln.set_block_size("x", 96);
                             kernel_soln.set_block_size("y", 28);
                             kernel_soln.set_block_size("z", 96);
                         }
                     }
                     );
            }
        }
    
    };

    // Create an object of type 'Iso3dfdStencil',
    // making it available in the YASK compiler utility via the
    // '-stencil' commmand-line option or the 'stencil=' build option.
    static Iso3dfdStencil Iso3dfdStencil_instance;

    // Add a sponge absorption factor.
    class Iso3dfdSpongeStencil : public Iso3dfdStencil {
    protected:

    public:
        Iso3dfdSpongeStencil(int radius=8) :
            Iso3dfdStencil("_sponge", radius) { }
        virtual ~Iso3dfdSpongeStencil() { }

        // Define equation for p at t+1 based on values from v and p at t.
        virtual void define() override {

            // Sponge coefficients.
            // In practice, the interior values would be set to 1.0,
            // and values nearer the boundary would be set to values
            // increasingly approaching 0.0.
            yc_var_proxy cr_x("cr_x", get_soln(), { x });
            yc_var_proxy cr_y("cr_y", get_soln(), { y });
            yc_var_proxy cr_z("cr_z", get_soln(), { z });
        
            // Get equation for RHS.
            auto next_p = get_next_p();

            // Apply sponge absorption.
            next_p *= cr_x(x) * cr_y(y) * cr_z(z);

            // Define the value at t+1 to be equal to next_p.
            p(t+1, x, y, z) EQUALS next_p;

            // Apply BKCs.
            set_configs();
        }
    };

    // Create an object of type 'Iso3dfdSpongeStencil',
    // making it available in the YASK compiler utility via the
    // '-stencil' commmand-line option or the 'stencil=' build option.
    static Iso3dfdSpongeStencil Iso3dfdSpongeStencil_instance;

} // namespace.
