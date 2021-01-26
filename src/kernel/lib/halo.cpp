/*****************************************************************************

YASK: Yet Another Stencil Kit
Copyright (c) 2014-2020, Intel Corporation

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

// This file contains implementations halo exchange methods.
// Also see context.cpp, setup.cpp, and soln_apis.cpp.

#include "yask_stencil.hpp"
using namespace std;

namespace yask {

    /*
      Host halo exchange w/explicit shared memory (shm):
      rank I                              rank J (neighbor of I)
      ---------------------               ----------------------
      var A ----------------> shared_buf ---------------> var A
                 pack                          unpack

      Host halo exchange w/o shm:
      rank I                              rank J (neighbor of I)
      ---------------------               ----------------------
      var A ----> local_buf ------------> local_buf ----> var A
             pack        MPI_isend   MPI_irecv     unpack
             (implicit shm created by MPI lib not shown)

      Device halo exchange w/o direct device copy and w/o shm:
      rank I                              rank J (neighbor of I)
      -----------------------             -----------------------
      var A --> dev_local_buf             dev_local_buf --> var A
            pack      | copy to host            ^     unpack
                      V                         | copy to dev
               host_local_buf ---------> host_local_buf
                         MPI_isend   MPI_irecv

      Device halo exchange w/o direct device copy and w/shm:
      --not implemented [yet].

      Device halo exchange w/direct device copy:
      --not implemented [yet].
    */
    
    // Exchange dirty halo data for all vars and all steps.
    void StencilContext::exchange_halos() {

        #if defined(USE_MPI) && !defined(NO_HALO_EXCHANGE)
        STATE_VARS(this);
        if (!enable_halo_exchange || env->num_ranks < 2)
            return;
        auto& use_offload = KernelEnv::_use_offload;

        halo_time.start();
        double wait_delta = 0.;
        TRACE_MSG("exchange_halos");
        if (is_overlap_active()) {
            if (do_mpi_left)
                TRACE_MSG(" following calc of MPI left exterior");
            if (do_mpi_right)
                TRACE_MSG(" following calc of MPI right exterior");
            if (do_mpi_interior)
                TRACE_MSG(" following calc of MPI interior");
        }

        // Vars for list of vars that need to be swapped and their step
        // indices.  Use an ordered map by *name* to make sure vars are
        // swapped in same order on all ranks. (If we order vars by
        // pointer, pointer values will not generally be the same on each
        // rank.)
        VarPtrMap vars_to_swap;
        map<YkVarPtr, idx_t> first_steps_to_swap;
        map<YkVarPtr, idx_t> last_steps_to_swap;

        // Loop thru all vars in stencil.
        for (auto& gp : orig_var_ptrs) {
            auto& gb = gp->gb();

            // Don't swap scratch vars.
            if (gb.is_scratch())
                continue;

            // Only need to swap data in vars that have any MPI buffers.
            auto& gname = gp->get_name();
            if (mpi_data.count(gname) == 0)
                continue;

            // Check all allocated step indices.
            // Use '0' for vars that don't use the step dim.
            idx_t start_t = 0, stop_t = 1;
            if (gp->is_dim_used(step_dim)) {
                start_t = gp->get_first_valid_step_index();
                stop_t = gp->get_last_valid_step_index() + 1;
            }
            for (idx_t t = start_t; t < stop_t; t++) {

                // Only need to swap vars whose halos are not up-to-date
                // for this step.
                if (!gb.is_dirty(t))
                    continue;

                // Swap this var.
                vars_to_swap[gname] = gp;

                // Update first step.
                if (first_steps_to_swap.count(gp) == 0 || t < first_steps_to_swap[gp])
                    first_steps_to_swap[gp] = t;

                // Update last step.
                if (last_steps_to_swap.count(gp) == 0 || t > last_steps_to_swap[gp])
                    last_steps_to_swap[gp] = t;

            } // steps.
        } // vars.
        TRACE_MSG("exchange_halos: need to exchange halos for " <<
                  vars_to_swap.size() << " var(s)");
        assert(vars_to_swap.size() == first_steps_to_swap.size());
        assert(vars_to_swap.size() == last_steps_to_swap.size());

        // Sequence of things to do for each neighbor.
        enum halo_steps { halo_irecv, halo_pack_isend, halo_unpack, halo_final };
        vector<halo_steps> steps_to_do;

        // Flags indicate what part of vars were most recently calc'd.
        // These determine what exchange steps need to be done now.
        if (vars_to_swap.size()) {
            if (do_mpi_left || do_mpi_right) {
                steps_to_do.push_back(halo_irecv);
                steps_to_do.push_back(halo_pack_isend);
            }
            if (do_mpi_interior) {
                steps_to_do.push_back(halo_unpack);
                steps_to_do.push_back(halo_final);
            }
        }

        int num_send_reqs = 0;
        int num_recv_reqs = 0;
        for (auto halo_step : steps_to_do) {

            if (halo_step == halo_irecv)
                TRACE_MSG("exchange_halos: requesting data phase");
            else if (halo_step == halo_pack_isend)
                TRACE_MSG("exchange_halos: packing and sending data phase");
            else if (halo_step == halo_unpack)
                TRACE_MSG("exchange_halos: waiting for and unpacking data phase");
            else if (halo_step == halo_final)
                TRACE_MSG("exchange_halos: waiting for send to finish phase");
            else
                THROW_YASK_EXCEPTION("internal error: unknown halo-exchange step");

            // Loop thru all vars to swap.
            // Use 'gi' as an MPI tag.
            int gi = 0;
            for (auto gtsi : vars_to_swap) {
                gi++;
                auto& gname = gtsi.first;
                auto& gp = gtsi.second;
                auto& gb = gp->gb();
                auto& var_mpi_data = mpi_data.at(gname);
                MPI_Request* var_recv_reqs = var_mpi_data.recv_reqs.data();
                MPI_Request* var_send_reqs = var_mpi_data.send_reqs.data();

                // Loop thru all this rank's neighbors.
                var_mpi_data.visit_neighbors
                    ([&](const IdxTuple& offsets, // NeighborOffset.
                         int neighbor_rank,
                         int ni, // unique neighbor index.
                         MPIBufs& bufs) {
                         auto& send_buf = bufs.bufs[MPIBufs::buf_send];
                         auto& recv_buf = bufs.bufs[MPIBufs::buf_recv];
                         TRACE_MSG("exchange_halos:   with rank " << neighbor_rank << " at relative position " <<
                                   offsets.sub_elements(1).make_dim_val_offset_str());

                         // Are we using MPI shm w/this neighbor?
                         bool using_shm = opts->use_shm && mpi_info->shm_ranks.at(ni) != MPI_PROC_NULL;

                         // Submit async request to receive data from neighbor.
                         if (halo_step == halo_irecv) {
                             auto nbytes = recv_buf.get_bytes();
                             if (nbytes) {
                                 if (using_shm)
                                     TRACE_MSG("exchange_halos:    no receive req due to shm");
                                 else {
                                     void* buf = (void*)recv_buf._elems;
                                     TRACE_MSG("exchange_halos:    requesting up to " << make_byte_str(nbytes));
                                     auto& r = var_recv_reqs[ni];
                                     MPI_Irecv(buf, nbytes, MPI_BYTE,
                                               neighbor_rank, int(gi),
                                               env->comm, &r);
                                     num_recv_reqs++;
                                 }
                             }
                             else
                                 TRACE_MSG("exchange_halos:    0B to request");
                         }

                         // Pack data into send buffer, then send to neighbor.
                         else if (halo_step == halo_pack_isend) {
                             auto nbytes = send_buf.get_bytes();
                             if (nbytes) {

                                 // Vec ok?
                                 // Domain sizes must be ok, and buffer size must be ok
                                 // as calculated when buffers were created.
                                 bool send_vec_ok = send_buf.vec_copy_ok;

                                 // Get first and last indices to pack from.
                                 IdxTuple first = send_buf.begin_pt;
                                 IdxTuple last = send_buf.last_pt;

                                 // The code in alloc_mpi_data() pre-calculated the first and
                                 // last points of each buffer, except in the step dim, where
                                 // the max range was set. Update actual range now.
                                 if (gp->is_dim_used(step_dim)) {
                                     first.set_val(step_dim, first_steps_to_swap[gp]);
                                     last.set_val(step_dim, last_steps_to_swap[gp]);
                                 }

                                 // Wait until buffer is avail if sharing one.
                                 if (using_shm) {
                                     TRACE_MSG("exchange_halos:    waiting to write to shm buffer");
                                     halo_wait_time.start();
                                     send_buf.wait_for_ok_to_write();
                                     wait_delta += halo_wait_time.stop();
                                 }

                                 // Copy (pack) data from var to buffer.
                                 void* buf = (void*)send_buf._elems;
                                 TRACE_MSG("exchange_halos:    packing [" << first.make_dim_val_str() <<
                                           " ... " << last.make_dim_val_str() << "] " <<
                                           (send_vec_ok ? "with" : "without") <<
                                           " vector copy into " << buf <<
                                           (use_offload ? " on device" : " on host"));
                                 idx_t nelems = 0;
                                 halo_pack_time.start();
                                 if (send_vec_ok)
                                     nelems = gb.get_vecs_in_slice(buf, first, last, use_offload);
                                 else
                                     nelems = gb.get_elements_in_slice(buf, first, last, use_offload);
                                 halo_pack_time.stop();
                                 idx_t nbytes = nelems * get_element_bytes();

                                 if (use_offload) {
                                     TRACE_MSG("exchange_halos:    copying buffer from device");
                                     halo_copy_time.start();
                                     offload_copy_from_device(buf, nbytes);
                                     halo_copy_time.stop();
                                 }

                                 if (using_shm) {
                                     TRACE_MSG("exchange_halos:    no send req due to shm");
                                     send_buf.mark_write_done();
                                 }
                                 else {

                                     // Send packed buffer to neighbor.
                                     assert(nbytes <= send_buf.get_bytes());
                                     TRACE_MSG("exchange_halos:    sending " << make_byte_str(nbytes));
                                     auto& r = var_send_reqs[ni];
                                     MPI_Isend(buf, nbytes, MPI_BYTE,
                                               neighbor_rank, int(gi), env->comm, &r);
                                     num_send_reqs++;
                                 }
                             }
                             else
                                 TRACE_MSG("   0B to send");
                         }

                         // Wait for data from neighbor, then unpack it.
                         else if (halo_step == halo_unpack) {
                             auto nbytes = recv_buf.get_bytes();
                             if (nbytes) {

                                 // Wait until data in buffer is avail.
                                 if (using_shm) {
                                     TRACE_MSG("exchange_halos:    waiting for data in shm buffer");
                                     halo_wait_time.start();
                                     recv_buf.wait_for_ok_to_read();
                                     wait_delta += halo_wait_time.stop();
                                 }
                                 else {

                                     // Wait for data from neighbor before unpacking it.
                                     auto& r = var_recv_reqs[ni];
                                     if (r != MPI_REQUEST_NULL) {
                                         TRACE_MSG("exchange_halos:    waiting for receipt of " <<
                                                   make_byte_str(nbytes));
                                         halo_wait_time.start();
                                         MPI_Wait(&r, MPI_STATUS_IGNORE);
                                         wait_delta += halo_wait_time.stop();
                                     }
                                     r = MPI_REQUEST_NULL;
                                 }

                                 // Vec ok?
                                 bool recv_vec_ok = recv_buf.vec_copy_ok;

                                 // Get first and last ranges.
                                 IdxTuple first = recv_buf.begin_pt;
                                 IdxTuple last = recv_buf.last_pt;

                                 // Set step val as above.
                                 if (gp->is_dim_used(step_dim)) {
                                     first.set_val(step_dim, first_steps_to_swap[gp]);
                                     last.set_val(step_dim, last_steps_to_swap[gp]);
                                 }

                                 void* buf = (void*)recv_buf._elems;
                                 if (use_offload) {
                                     TRACE_MSG("exchange_halos:    copying buffer to device");
                                     halo_copy_time.start();
                                     offload_copy_to_device(buf, nbytes);
                                     halo_copy_time.stop();
                                 }

                                 // Copy data from buffer to var.
                                 TRACE_MSG("exchange_halos:    got data; unpacking into [" <<
                                           first.make_dim_val_str() <<
                                           " ... " << last.make_dim_val_str() << "] " <<
                                           (recv_vec_ok ? "with" : "without") <<
                                           " vector copy from " << buf <<
                                           (use_offload ? " on device" : " on host"));
                                 idx_t nelems = 0;
                                 halo_unpack_time.start();
                                 if (recv_vec_ok)
                                     nelems = gp->set_vecs_in_slice(buf, first, last, use_offload);
                                 else
                                     nelems = gp->set_elements_in_slice(buf, first, last, use_offload);
                                 halo_unpack_time.stop();
                                 assert(nelems <= recv_buf.get_size());

                                 if (using_shm)
                                     recv_buf.mark_read_done();
                             }
                             else
                                 TRACE_MSG("exchange_halos:    0B to wait for");
                         }

                         // Final steps.
                         else if (halo_step == halo_final) {
                             auto nbytes = send_buf.get_bytes();
                             if (nbytes) {

                                 if (using_shm)
                                     TRACE_MSG("exchange_halos:    no send wait due to shm");
                                 else {

                                     // Wait for send to finish.
                                     // TODO: consider using MPI_WaitAll.
                                     // TODO: strictly, we don't have to wait on the
                                     // send to finish until we want to reuse this buffer,
                                     // so we could wait on the *previous* send right before
                                     // doing another one.
                                     auto& r = var_send_reqs[ni];
                                     if (r != MPI_REQUEST_NULL) {
                                         TRACE_MSG("   waiting to finish send of " << make_byte_str(nbytes));
                                         halo_wait_time.start();
                                         MPI_Wait(&var_send_reqs[ni], MPI_STATUS_IGNORE);
                                         wait_delta += halo_wait_time.stop();
                                     }
                                     r = MPI_REQUEST_NULL;
                                 }
                             }

                             // Mark vars as up-to-date when done.
                             for (idx_t si = first_steps_to_swap[gp]; si <= last_steps_to_swap[gp]; si++) {
                                 if (gb.is_dirty(si)) {
                                     gb.set_dirty(false, si);
                                     TRACE_MSG("exchange_halos: var '" << gname <<
                                               "' marked as clean at step-index " << si);
                                 }
                             }
                         }

                     }); // visit neighbors.

            } // vars.

        } // exchange sequence.

        TRACE_MSG("exchange_halos: " << num_recv_reqs << " MPI receive request(s) issued");
        TRACE_MSG("exchange_halos: " << num_send_reqs << " MPI send request(s) issued");
        auto mpi_call_time = halo_time.stop();
        TRACE_MSG("exchange_halos: secs spent in MPI waits: " << make_num_str(wait_delta));
        TRACE_MSG("exchange_halos: secs spent in this call: " << make_num_str(mpi_call_time));
        #endif
    }

    // Call MPI_Test() on all unfinished requests to promote MPI progress.
    // TODO: replace with more direct and less intrusive techniques.
    void StencilContext::poke_halo_exchange() {

        #if defined(USE_MPI) && !defined(NO_HALO_EXCHANGE)
        STATE_VARS(this);
        if (!enable_halo_exchange || env->num_ranks < 2)
            return;

        halo_test_time.start();
        TRACE_MSG("poke_halo_exchange");

        // Loop thru MPI data.
        int num_tests = 0;
        for (auto& mdi : mpi_data) {
            auto& gname = mdi.first;
            auto& var_mpi_data = mdi.second;
            MPI_Request* var_recv_reqs = var_mpi_data.recv_reqs.data();
            MPI_Request* var_send_reqs = var_mpi_data.send_reqs.data();

            int flag;
            #if 1
            int indices[max(var_mpi_data.recv_reqs.size(), var_mpi_data.send_reqs.size())];
            MPI_Testsome(int(var_mpi_data.recv_reqs.size()), var_recv_reqs, &flag, indices, MPI_STATUS_IGNORE);
            MPI_Testsome(int(var_mpi_data.send_reqs.size()), var_send_reqs, &flag, indices, MPI_STATUS_IGNORE);
            #elif 0
            int index;
            MPI_Testany(int(var_mpi_data.recv_reqs.size()), var_recv_reqs, &index, &flag, MPI_STATUS_IGNORE);
            MPI_Testany(int(var_mpi_data.send_reqs.size()), var_send_reqs, &index, &flag, MPI_STATUS_IGNORE);
            #else
            for (size_t i = 0; i < var_mpi_data.recv_reqs.size(); i++) {
                auto& r = var_recv_reqs[i];
                if (r != MPI_REQUEST_NULL) {
                    //TRACE_MSG(gname << " recv test &MPI_Request = " << &r);
                    MPI_Test(&r, &flag, MPI_STATUS_IGNORE);
                    num_tests++;
                    if (flag)
                        r = MPI_REQUEST_NULL;
                }
            }
            for (size_t i = 0; i < var_mpi_data.send_reqs.size(); i++) {
                auto& r = var_send_reqs[i];
                if (r != MPI_REQUEST_NULL) {
                    //TRACE_MSG(gname << " send test &MPI_Request = " << &r);
                    MPI_Test(&r, &flag, MPI_STATUS_IGNORE);
                    num_tests++;
                    if (flag)
                        r = MPI_REQUEST_NULL;
                }
            }
            #endif
        }
        auto ttime = halo_test_time.stop();
        TRACE_MSG("poke_halo_exchange: secs spent in " << num_tests <<
                  " MPI test(s): " << make_num_str(ttime));
        #endif
    }
    
} // namespace yask.
