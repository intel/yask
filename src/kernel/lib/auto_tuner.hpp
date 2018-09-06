/*****************************************************************************

YASK: Yet Another Stencil Kernel
Copyright (c) 2014-2018, Intel Corporation

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

#pragma once

namespace yask {

    // Auto-tuner for setting block size.
    class AutoTuner {
    protected:
        StencilContext* _context = 0;
        KernelSettings* _settings = 0; // settings to change.
        std::string _name;

        // Null stream to throw away debug info.
        yask_output_factory yof;
        yask_output_ptr nullop = yof.new_null_output();
        
        // Whether to print progress.
        bool verbose = false;

        // AT parameters.
        double warmup_steps = 10;
        double warmup_secs = 0.5;
        idx_t min_steps = 10;
        double min_secs = 0.25; // eval when either min_steps or min_secs is reached.
        idx_t min_step = 4;
        idx_t max_radius = 64;
        idx_t min_pts = 512; // 8^3.
        idx_t min_blks = 4;

        // Results.
        std::map<IdxTuple, double> results; // block-size -> perf.
        int n2big = 0, n2small = 0;

        // Best so far.
        IdxTuple best_block;
        double best_rate = 0.;

        // Current point in search.
        IdxTuple center_block;
        idx_t radius = 0;
        bool done = false;
        idx_t neigh_idx = 0;
        bool better_neigh_found = false;

        // Cumulative vars.
        double ctime = 0.;
        idx_t csteps = 0;
        bool in_warmup = true;

    public:
        static constexpr idx_t max_step_t = 4;

        AutoTuner(StencilContext* ctx,
                  KernelSettings* settings,
                  const std::string& name = "") :
            _context(ctx),
            _settings(settings) {
            _name = "auto-tuner";
            if (name.length())
                _name += "(" + name + ")";
        }

        // Start & stop this timer to track elapsed time.
        YaskTimer timer;

        // Change settings pointer.
        void set_settings(KernelSettings* p) {
            _settings = p;
        }
        
        // Reset all state to beginning.
        void clear(bool mark_done, bool verbose = false);

        // Evaluate the previous run and take next auto-tuner step.
        void eval(idx_t steps);

        // Print the best settings.
        void print_settings(std::ostream& os) const;

        // Apply settings.
        void apply();

        // Done?
        bool is_done() const { return done; }
    };

} // yask namespace.
