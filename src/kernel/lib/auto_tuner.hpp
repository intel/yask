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

#pragma once

namespace yask {

    // Auto-tuner for setting block size.
    class AutoTuner :
        public ContextLinker {

    protected:

        // Settings to change. May be pointer to solution settings
        // or local settings for a stage.
        KernelSettings* _settings = 0;

        // Name of tuner.
        std::string _name;

        // String to print before each msg.
        std::string _prefix;
        #define AT_DEBUG_MSG(msg) DEBUG_MSG(_prefix << msg)
        #define AT_TRACE_MSG(msg) TRACE_MSG(_prefix << msg)

        // Null stream to throw away debug info.
        yask_output_factory yof;
        yask_output_ptr nullop = yof.new_null_output();

        // Whether to print progress.
        bool verbose = false;

        // AT parameters.
        double warmup_steps = 100;
        double warmup_secs = 0.5; // end warmup when either warmup_steps OR warmup_secs is reached.
        idx_t trial_steps = 100;
        double trial_secs = 0.25; // end trial when either trial_steps OR trial_secs is reached.
        double cutoff = 0.8;   // can stop eval if current rate < best rate * cutoff;
        idx_t min_dist = 4;     // min distance to move in any direction per trial.
        idx_t min_pts = 512; // 8^3; min points in a block.
        idx_t min_blks = 1;  // min number of blocks.
        idx_t max_radius = 8;   // starting search radius.

        // State of the search of one set of target sizes.
        struct AutoTunerState {

            // Results.
            std::unordered_map<IdxTuple, double> results; // block-size -> perf cache.
            int n2big = 0, n2small = 0, n2far = 0;

            // Best so far.
            IdxTuple best_sizes;
            double best_rate = 0.;

            // Current location of search.
            IdxTuple center_sizes;
            idx_t radius = 0;
            idx_t neigh_idx = 0;
            bool better_neigh_found = false;

            // Cumulative data within a trial or warmup.
            double ctime = 0.;
            idx_t csteps = 0;
            bool in_warmup = true;

            void init(AutoTuner* at, bool warmup_needed) {
                results.clear();
                n2big = n2small = n2far = 0;
                best_sizes.set_vals_same(0);
                best_rate = 0.;
                center_sizes.set_vals_same(0);
                radius = at->max_radius;
                neigh_idx = 0;
                better_neigh_found = false;
                ctime = 0.;
                csteps = 0;
                in_warmup = warmup_needed;
            }
        };

        // Current state of search.
        AutoTunerState at_state;
        bool done = false;
        IdxTuple* outerp = 0;
        IdxTuple* targetp = 0;
        idx_t target_steps = 0;
 
        bool check_sizes(const IdxTuple& sizes);
        bool next_target();

    public:
        static constexpr idx_t max_stride_t = 4;

        AutoTuner(StencilContext* context,
                  KernelSettings* settings,
                  const std::string& name = "");
        virtual ~AutoTuner() {}

        // Start & stop this timer to track elapsed time.
        YaskTimer timer;

        // Increment this to track steps.
        idx_t steps_done = 0;

        // Reset all state to beginning.
        void clear(bool mark_done, bool verbose = false);

        // Evaluate the previous run and take next auto-tuner step.
        void eval();

        // Print the best settings.
        void print_settings() const;
        void print_temporal_settings() const;

        // If 'use_best', apply best settings and other kernel settings.
        // Else, just set other kernel settings.
        void apply(bool use_best);

        // Done?
        bool is_done() const { return done; }
    };

} // yask namespace.
