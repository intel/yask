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

// Purpose: implement a simple,infinite-size cache model to check
// prefetch and/or eviction coverage.

#pragma once

#include <map>

namespace yask {

    // key is CL addr; value is line num.
    class Cache : public std::map<uintptr_t, int> {
        int my_level;
        size_t num_reads, num_pfs, num_evicts, num_low_pfs;
        size_t num_reads_not_pfed, num_extra_pfs, num_bad_evicts, num_low_pfs_not_pfed;
        size_t num_sizes, sum_sizes, max_size;
        static const int max_msgs = 20;
        bool enabled;
        uintptr_t prev_read_line;
        std::map<intptr_t, size_t> stride_counts;
        static const float min_stride_pct = 0.1f;

    public:
        Cache(int l): my_level(l), num_reads(0), num_pfs(0), num_evicts(0),
                      num_reads_not_pfed(0), num_extra_pfs(0), num_bad_evicts(0),
                      num_sizes(0), sum_sizes(0), max_size(0), enabled(true),
                      prev_read_line(0)
        {
            printf("modeling cache L%i (a max of %i warnings of each type will be printed)...\n",
                   my_level, max_msgs);
        }

        void disable() { enabled = false; }
        void enable() { enabled = true; }
        bool is_enabled() const { return enabled; }

        static float lines2_mb(size_t l) {
            return (float)l * CACHELINE_BYTES / (1024*1024);
        }

        void dump_stats() const {
            int np = 0, nr = 0;
            for (auto const& i : *this) {
                void *p = (void*)(i.first * CACHELINE_BYTES);
                int line = i.second;
                if (line >= 0 && ++np < max_msgs)
                    printf("cache L%i: prefetch of %p from line %i without any read.\n", my_level,
                           p, line);
                if (line < 0 && ++nr < max_msgs)
                    printf("cache L%i: read of %p from line %i without any eviction.\n", my_level,
                           p, -line);
            }
            if (num_sizes)
                printf("cache L%i stats:\n"
                       " cur num elements = %lli lines (%.4f MB).\n"
                       " max num elements = %lli lines (%.4f MB).\n"
                       " ave num elements = %lli lines (%.4f MB).\n", my_level,
                       size(), lines2_mb(size()),
                       max_size, lines2_mb(max_size),
                       sum_sizes / num_sizes, lines2_mb(sum_sizes) / num_sizes);
            printf(" num reads = %lli.\n", num_reads);
            printf("  num reads of missing lines = %lli.%s\n", num_reads_not_pfed,
                   num_reads_not_pfed? " <<<<< WARNING" : "");
            printf("  num lines read but never evicted = %i.\n", nr);
            printf(" num prefetches = %lli.\n", num_pfs);
            printf("  num prefetches of lines never subsequently read = %i.\n", np);
            printf("  num prefetches of lines already in cache = %lli.\n", num_extra_pfs);
            printf(" num evictitions = %lli.\n", num_evicts);
            printf("  num evictions to non-existant lines = %lli.\n", num_bad_evicts);
            if (my_level > 1) {
                printf(" num prefetches into L%i = %lli.\n", my_level-1, num_low_pfs);
                printf("  num prefetches into L%i of missing lines = %lli.%s\n", my_level-1, num_low_pfs_not_pfed,
                       num_low_pfs_not_pfed? " <<<<< WARNING!" : "");
            }
            if (num_reads > 1 && stride_counts.size()) {
                printf(" stride counts (>= %f%% of num reads):\n", min_stride_pct);
                for (auto& i : stride_counts) {
                    float p = 100.f * (float)i.second / (float)num_reads;
                    if (p >= min_stride_pct)
                        printf("  %lli: %llu (%f%%)\n", i.first, i.second, p);
                }
            }
        }

        void update_size() {
            size_t s = size();
            if (s > max_size)
                max_size = s;

            // keep stats for calculating ave.
            sum_sizes += s;
            num_sizes++;
        }

        void evict(const void* p, int hint, int line) {
            if (!enabled) return;

            // eviction from lower caches ignored.
            // NB: this code assumes hint corresponds 1:1 to level.
            if (hint < my_level) return;
            uintptr_t k = (uintptr_t)p / CACHELINE_BYTES;

#pragma omp critical
            {
                // only report bad evicts if level matches.
                if (hint == my_level && count(k) == 0) {
                    if (++num_bad_evicts < max_msgs)
                        printf("cache L%i: eviction of non-existant %p at line %i\n", my_level, p, line);
                }

                erase(k);
                num_evicts++;
                update_size();
            }
        }

        void prefetch(const void* p, int hint, int line) {
            if (!enabled) return;
            uintptr_t k = (uintptr_t)p / CACHELINE_BYTES;

            // prefetch higher cache.
            if (hint > my_level) return;

#pragma omp critical
            {
                // prefetch this cache.
                // NB: this code assumes hint corresponds 1:1 to level.
                if (hint == my_level) {
                    if (count(k) > 0) {
                        if (++num_extra_pfs < max_msgs) {
                            printf("cache L%i: redundant prefetch of %p at line %i", my_level, p, line);
                            int old_line = operator[](k);
                            if (old_line < 0)
                                printf(" after a read at line %i.\n", -old_line);
                            else
                                printf(" after a prefetch at line %i.\n", old_line);
                        }
                    }
                }

                // prefetch closer cache.
                else {
                    if (count(k) == 0) {
                        if (++num_low_pfs_not_pfed < max_msgs)
                            printf("cache L%i: L%i prefetch of non-existant %p at line %i\n", my_level, hint, p, line);
                    }
                    num_low_pfs++;
                }

                // mark lines prefetched whether they were from this cache or a closer one.
                operator[](k) = line; // remember lines for later report.
                num_pfs++;
                update_size();
            }
        }

        void read(const void* p, int line) {
            if (!enabled) return;
            uintptr_t k = (uintptr_t)p / CACHELINE_BYTES;

#pragma omp critical
            {
                if (count(k) == 0) {
                    if (++num_reads_not_pfed < max_msgs)
                        printf("cache L%i: read of non-existant %p at line %i\n", my_level, p, line);
                }
                operator[](k) = -line; // neg line indicates read.

#ifdef TRACK_STRIDES
                // track stride.
                if (prev_read_line) {
                    intptr_t stride = (intptr_t)k - (intptr_t)prev_read_line;
                    stride_counts[stride]++;
                }
#endif

                prev_read_line = k;
                num_reads++;
                update_size();
            }
        }

        // Currently not monitoring writes.
        void write(const void* p, int line) {
        }
    };

}
