/*****************************************************************************

YASK: Yet Another Stencil Kernel
Copyright (c) 2014-2016, Intel Corporation

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

#include <map>

namespace yask {

    // key is CL addr; value is line num.
    class Cache : public map<uintptr_t, int> {
        int myLevel;
        size_t numReads, numPFs, numEvicts, numLowPFs;
        size_t numReadsNotPFed, numExtraPFs, numBadEvicts, numLowPFsNotPFed;
        size_t numSizes, sumSizes, maxSize;
        static const int maxMsgs = 20;
        bool enabled;
        uintptr_t prevReadLine;
        map<intptr_t, size_t> strideCounts;
        static const float minStridePct = 0.1f;

    public:
        Cache(int l): myLevel(l), numReads(0), numPFs(0), numEvicts(0),
                      numReadsNotPFed(0), numExtraPFs(0), numBadEvicts(0),
                      numSizes(0), sumSizes(0), maxSize(0), enabled(true),
                      prevReadLine(0)
        {
            printf("modeling cache L%i (a max of %i warnings of each type will be printed)...\n",
                   myLevel, maxMsgs);
        }

        void disable() { enabled = false; }
        void enable() { enabled = true; }
        bool isEnabled() const { return enabled; }

        static float lines2MB(size_t l) {
            return (float)l * CACHELINE_BYTES / (1024*1024);
        }

        void dumpStats() const {
            int np = 0, nr = 0;
            for (const_iterator i = begin(); i != end(); i++) {
                void *p = (void*)(i->first * CACHELINE_BYTES);
                int line = i->second;
                if (line >= 0 && ++np < maxMsgs)
                    printf("cache L%i: prefetch of %p from line %i without any read.\n", myLevel,
                           p, line);
                if (line < 0 && ++nr < maxMsgs)
                    printf("cache L%i: read of %p from line %i without any eviction.\n", myLevel,
                           p, -line);
            }
            if (numSizes)
                printf("cache L%i stats:\n"
                       " cur num elements = %lli lines (%.4f MB).\n"
                       " max num elements = %lli lines (%.4f MB).\n"
                       " ave num elements = %lli lines (%.4f MB).\n", myLevel,
                       size(), lines2MB(size()),
                       maxSize, lines2MB(maxSize), 
                       sumSizes / numSizes, lines2MB(sumSizes) / numSizes);
            printf(" num reads = %lli.\n", numReads);
            printf("  num reads of missing lines = %lli.%s\n", numReadsNotPFed,
                   numReadsNotPFed? " <<<<< WARNING" : "");
            printf("  num lines read but never evicted = %i.\n", nr);
            printf(" num prefetches = %lli.\n", numPFs);
            printf("  num prefetches of lines never subsequently read = %i.\n", np);
            printf("  num prefetches of lines already in cache = %lli.\n", numExtraPFs);
            printf(" num evictitions = %lli.\n", numEvicts);
            printf("  num evictions to non-existant lines = %lli.\n", numBadEvicts);
            if (myLevel > 1) {
                printf(" num prefetches into L%i = %lli.\n", myLevel-1, numLowPFs);
                printf("  num prefetches into L%i of missing lines = %lli.%s\n", myLevel-1, numLowPFsNotPFed,
                       numLowPFsNotPFed? " <<<<< WARNING!" : "");
            }
            if (numReads > 1 && strideCounts.size()) {
                printf(" stride counts (>= %f%% of num reads):\n", minStridePct);
                for (auto& i : strideCounts) {
                    float p = 100.f * (float)i.second / (float)numReads;
                    if (p >= minStridePct)
                        printf("  %lli: %llu (%f%%)\n", i.first, i.second, p);
                }
            }
        }

        void updateSize() {
            size_t s = size();
            if (s > maxSize) 
                maxSize = s;

            // keep stats for calculating ave.
            sumSizes += s;
            numSizes++;
        }

        void evict(const void* p, int hint, int line) {
            if (!enabled) return;

            // eviction from lower caches ignored. 
            // FIXME: this code assumes hint corresponds 1:1 to level.
            if (hint < myLevel) return;
            uintptr_t k = (uintptr_t)p / CACHELINE_BYTES;

#pragma omp critical
            {
                // only report bad evicts if level matches.
                if (hint == myLevel && count(k) == 0) {
                    if (++numBadEvicts < maxMsgs)
                        printf("cache L%i: eviction of non-existant %p at line %i\n", myLevel, p, line);
                }

                erase(k);
                numEvicts++;
                updateSize();
            }
        }

        void prefetch(const void* p, int hint, int line) {
            if (!enabled) return;
            uintptr_t k = (uintptr_t)p / CACHELINE_BYTES;

            // prefetch higher cache.
            if (hint > myLevel) return;

#pragma omp critical
            {
                // prefetch this cache.
                // FIXME: this code assumes hint corresponds 1:1 to level.
                if (hint == myLevel) {
                    if (count(k) > 0) {
                        if (++numExtraPFs < maxMsgs) {
                            printf("cache L%i: redundant prefetch of %p at line %i", myLevel, p, line);
                            int oldLine = operator[](k);
                            if (oldLine < 0)
                                printf(" after a read at line %i.\n", -oldLine);
                            else
                                printf(" after a prefetch at line %i.\n", oldLine);
                        }
                    }
                }

                // prefetch closer cache.
                else {
                    if (count(k) == 0) {
                        if (++numLowPFsNotPFed < maxMsgs)
                            printf("cache L%i: L%i prefetch of non-existant %p at line %i\n", myLevel, hint, p, line);
                    }
                    numLowPFs++;
                }

                // mark lines prefetched whether they were from this cache or a closer one.
                operator[](k) = line; // remember lines for later report.
                numPFs++;
                updateSize();
            }
        }

        void read(const void* p, int line) {
            if (!enabled) return;
            uintptr_t k = (uintptr_t)p / CACHELINE_BYTES;

#pragma omp critical
            {
                if (count(k) == 0) {
                    if (++numReadsNotPFed < maxMsgs)
                        printf("cache L%i: read of non-existant %p at line %i\n", myLevel, p, line);
                }
                operator[](k) = -line; // neg line indicates read.

#ifdef TRACK_STRIDES
                // track stride.
                if (prevReadLine) {
                    intptr_t stride = (intptr_t)k - (intptr_t)prevReadLine;
                    strideCounts[stride]++;
                }
#endif

                prevReadLine = k;
                numReads++;
                updateSize();
            }
        }

        // Currently not monitoring writes.
        void write(const void* p, int line) {
        }
    };

}
