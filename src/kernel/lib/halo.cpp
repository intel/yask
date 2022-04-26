/*****************************************************************************

YASK: Yet Another Stencil Kit
Copyright (c) 2014-2022, Intel Corporation

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

      Device halo exchange w/o direct device copy w/o shm:
      rank I                              rank J (neighbor of I)
      -----------------------             -----------------------
      var A --> dev_local_buf             dev_local_buf --> var A
            pack      | copy to host            ^     unpack
                      V                         | copy to dev
               host_local_buf ---------> host_local_buf
                        MPI_isend   MPI_irecv

      Device halo exchange w/o direct device copy w/shm:
      --not implemented.

      Device halo exchange w/direct device copy w/o shm:
      rank I                             rank J (neighbor of I)
      ---------------------------        -----------------------------
      dev var A --> dev_local_buf -----> dev_local_buf ----> dev var A
               pack       MPI_isend    MPI_irecv      unpack
             (may still be implicitly routed through host.)
    */
    
    // Exchange dirty halo data for all vars and all steps.
    void StencilContext::exchange_halos() {

        #if defined(USE_MPI)
        STATE_VARS(this);
        if (!actl_opts->do_halo_exchange || env->num_ranks < 2)
            return;
        auto& use_offload = KernelEnv::_use_offload;
        auto use_device_mpi = use_offload ? actl_opts->use_device_mpi : false;

        halo_time.start();
        double wait_delta = 0.;
        if (is_overlap_active()) {
            if (do_mpi_left)
                TRACE_MSG("following calc of MPI left exterior");
            if (do_mpi_right)
                TRACE_MSG("following calc of MPI right exterior");
            if (do_mpi_interior)
                TRACE_MSG("following calc of MPI interior");
        }

        // Vars for list of vars that need to be swapped and their step
        // indices.
        struct SwapInfo {
            YkVarPtr gp;
            set<idx_t> steps;
        };
        vector<SwapInfo> vars_to_swap;

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
            bool first = true;
            for (idx_t t = start_t; t < stop_t; t++) {

                // Only need to swap vars whose halos are not up-to-date
                // for this step.
                if (!gb.is_dirty(t))
                    continue;

                // Swap this var.
                if (first) {
                    SwapInfo si;
                    si.gp = gp;
                    vars_to_swap.push_back(si);
                    first = false;
                }
                vars_to_swap.back().steps.insert(t);

            } // steps.
        } // vars.
        TRACE_MSG("need to exchange halos for " <<
                  vars_to_swap.size() << " var/step combo(s)");

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
                TRACE_MSG("requesting data phase");
            else if (halo_step == halo_pack_isend)
                TRACE_MSG("packing and sending data phase");
            else if (halo_step == halo_unpack)
                TRACE_MSG("waiting for and unpacking data phase");
            else if (halo_step == halo_final)
                TRACE_MSG("waiting for send to finish phase");
            else
                THROW_YASK_EXCEPTION("internal error: unknown halo-exchange step");

            // Loop thru all vars to swap.
            // Use 'gi' as an MPI tag.
            int gi = 0;
            for (auto& si : vars_to_swap) {
                gi++;
                auto gp = si.gp;
                auto& gb = gp->gb();
                auto& gname = gb.get_name();
                TRACE_MSG(" processing var '" << gname << "', " << si.steps.size() << " step(s)");
                auto& var_mpi_data = mpi_data.at(gname);
                auto* var_recv_reqs = var_mpi_data.recv_reqs.data();
                auto* var_send_reqs = var_mpi_data.send_reqs.data();
                auto* var_recv_stats = var_mpi_data.recv_stats.data();
                bool finalizing_var = false;

                // Loop thru all this rank's neighbors.
                var_mpi_data.visit_neighbors
                    ([&](const IdxTuple& offsets, // NeighborOffset.
                         int neighbor_rank,
                         int ni, // unique neighbor index.
                         MPIBufs& bufs) {
                         auto& send_buf = bufs.bufs[MPIBufs::buf_send];
                         auto& recv_buf = bufs.bufs[MPIBufs::buf_recv];
                         TRACE_MSG("exchange_halos:   with rank " << neighbor_rank <<
                                   " at relative position " <<
                                   offsets.sub_elements(1).make_dim_val_offset_str());

                         // Are we using MPI shm w/this neighbor?
                         bool using_shm = actl_opts->use_shm &&
                             mpi_info->shm_ranks.at(ni) != MPI_PROC_NULL;

                         // Submit async request to receive data from neighbor.
                         if (halo_step == halo_irecv) {
                             auto nbbytes = recv_buf.get_bytes();
                             if (nbbytes) {
                                 if (using_shm)
                                     TRACE_MSG("exchange_halos:    no receive req due to shm");
                                 else {
                                     void* buf = (void*)recv_buf._elems;
                                     void* rbuf = use_device_mpi ? get_dev_ptr(buf) : buf;
                                     TRACE_MSG("exchange_halos:    requesting up to " <<
                                               make_byte_str(nbbytes) << " into " << rbuf);
                                     auto& r = var_recv_reqs[ni];
                                     MPI_Irecv(rbuf, nbbytes, MPI_BYTE,
                                               neighbor_rank, int(gi),
                                               env->comm, &r);
                                     num_recv_reqs++;
                                 }
                             }
                             else
                                 TRACE_MSG("exchange_halos:    0B to request");
                         } // recv step.

                         // Pack data into send buffer, then send to neighbor.
                         else if (halo_step == halo_pack_isend) {
                             auto nbbytes = send_buf.get_bytes();
                             if (nbbytes) {

                                 // Vec ok?
                                 // Domain sizes must be ok, and buffer size must be ok
                                 // as calculated when buffers were created.
                                 bool send_vec_ok = send_buf.vec_copy_ok;

                                 // Get first and last indices to pack from.
                                 IdxTuple first = send_buf.begin_pt;
                                 IdxTuple last = send_buf.last_pt;

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
                                 size_t npbytes = 0;
                                 char* bufp = (char*)buf;
                                 
                                 // Pack one step at a time.
                                 halo_pack_time.start();
                                 for (auto t : si.steps) {
                                     if (gp->is_dim_used(step_dim)) {
                                         first.set_val(step_dim, t);
                                         last.set_val(step_dim, t);
                                     }
                                     idx_t nelems = 0;
                                     if (send_vec_ok)
                                         nelems = gb.get_vecs_in_slice(bufp, first, last, use_offload);
                                     else
                                         nelems = gb.get_elements_in_slice(bufp, first, last, use_offload);
                                     auto nb = nelems * get_element_bytes();
                                     bufp += nb;
                                     npbytes += nb;
                                 }
                                 halo_pack_time.stop();
                                 assert(npbytes <= nbbytes);

                                 if (use_offload && !use_device_mpi) {
                                     TRACE_MSG("exchange_halos:    copying buffer from device");
                                     halo_copy_time.start();
                                     offload_copy_from_device(buf, npbytes);
                                     halo_copy_time.stop();
                                     assert(!using_shm);
                                 }

                                 if (using_shm) {
                                     TRACE_MSG("exchange_halos:    put " << make_byte_str(npbytes) <<
                                               " into shm");
                                     send_buf.set_data(npbytes);  // Send size thru lock.
                                     send_buf.mark_write_done();
                                 }
                                 else {

                                     // Send packed buffer to neighbor.
                                     auto& r = var_send_reqs[ni];
                                     void* sbuf = use_device_mpi ? get_dev_ptr(buf) : buf;
                                     TRACE_MSG("exchange_halos:    sending " << make_byte_str(npbytes) <<
                                               " from " << sbuf);
                                     MPI_Isend(sbuf, npbytes, MPI_BYTE,
                                               neighbor_rank, int(gi), env->comm, &r);
                                     num_send_reqs++;
                                 }
                             }
                             else
                                 TRACE_MSG("   0B to send");
                         } // pack & send step.

                         // Wait for data from neighbor, then unpack it.
                         else if (halo_step == halo_unpack) {
                             auto nbbytes = recv_buf.get_bytes();
                             if (nbbytes) {
                                 int nbytes = 0;

                                 // Wait until data in buffer is avail.
                                 if (using_shm) {
                                     TRACE_MSG("exchange_halos:    waiting for data in shm buffer");
                                     halo_wait_time.start();
                                     recv_buf.wait_for_ok_to_read();
                                     wait_delta += halo_wait_time.stop();
                                     nbytes = recv_buf.get_data(); // Size was stored in lock.
                                 }
                                 else {

                                     auto& r = var_recv_reqs[ni];
                                     auto& s = var_recv_stats[ni];

                                     if (r == MPI_REQUEST_NULL) {
                                         // Already got status from an MPI_Test* or MPI_Wait* function.
                                         TRACE_MSG("exchange_halos:    already received of up to " <<
                                                   make_byte_str(nbbytes));
                                     }

                                     else {
                                         // Wait for data from neighbor before unpacking it.
                                         TRACE_MSG("exchange_halos:    waiting for receipt of up to " <<
                                                   make_byte_str(nbbytes));
                                         halo_wait_time.start();
                                         MPI_Wait(&r, &s);
                                         wait_delta += halo_wait_time.stop();
                                         r = MPI_REQUEST_NULL;
                                     }
                                     MPI_Get_count(&s, MPI_BYTE, &nbytes);
                                 }
                                 TRACE_MSG("exchange_halos:    got " << make_byte_str(nbytes));
                                 assert(nbytes <= nbbytes);

                                 if (!nbytes) {
                                     TRACE_MSG("exchange_halos:    received no data");
                                 } else {

                                     // Vec ok?
                                     bool recv_vec_ok = recv_buf.vec_copy_ok;

                                     // Get first and last ranges.
                                     IdxTuple first = recv_buf.begin_pt;
                                     IdxTuple last = recv_buf.last_pt;

                                     void* buf = (void*)recv_buf._elems;
                                     if (use_offload && !use_device_mpi) {
                                         TRACE_MSG("exchange_halos:    copying buffer to device");
                                         halo_copy_time.start();
                                         offload_copy_to_device(buf, nbytes);
                                         halo_copy_time.stop();
                                     }

                                     // Copy data from buffer to var.
                                     TRACE_MSG("exchange_halos:    unpacking into [" <<
                                               first.make_dim_val_str() <<
                                               " ... " << last.make_dim_val_str() << "] " <<
                                               (recv_vec_ok ? "with" : "without") <<
                                               " vector copy from " << buf <<
                                               (use_offload ? " on device" : " on host"));
                                     size_t npbytes = 0;
                                     char* bufp = (char*)buf;

                                     // Unpack one step at a time.                                 
                                     halo_unpack_time.start();
                                     for (auto t : si.steps) {
                                         if (gp->is_dim_used(step_dim)) {
                                             first.set_val(step_dim, t);
                                             last.set_val(step_dim, t);
                                         }
                                         idx_t nelems = 0;
                                         if (recv_vec_ok)
                                             nelems = gp->set_vecs_in_slice(bufp, first, last, use_offload);
                                         else
                                             nelems = gp->set_elements_in_slice(bufp, first, last, use_offload);
                                         auto nb = nelems * get_element_bytes();
                                         bufp += nb;
                                         npbytes += nb;
                                     }
                                     halo_unpack_time.stop();

                                     // Should have unpacked exactly what we got.
                                     assert(npbytes == nbytes);
                                 }
                                 
                                 if (using_shm)
                                     recv_buf.mark_read_done();
                             }
                             else
                                 TRACE_MSG("exchange_halos:    0B to wait for");
                         } // unpack step.

                         // Final steps.
                         else if (halo_step == halo_final) {
                             auto nbbytes = send_buf.get_bytes();
                             if (nbbytes) {

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
                                         TRACE_MSG("   waiting to finish send of up to " << make_byte_str(nbbytes));
                                         halo_wait_time.start();
                                         MPI_Wait(&var_send_reqs[ni], MPI_STATUS_IGNORE);
                                         wait_delta += halo_wait_time.stop();
                                     }
                                     r = MPI_REQUEST_NULL;
                                 }
                             }
                             finalizing_var = true;
                             
                         } // final step.

                     }); // visit neighbors.

                // Did we finish w/this var?
                if (finalizing_var) {

                    // Mark var as up-to-date.
                    for (auto t : si.steps) {
                        if (gb.is_dirty(t)) {
                            gb.set_dirty(false, t);
                            TRACE_MSG(" var '" << gname <<
                                      "' marked as clean at step-index " << t);
                        }
                        else
                            TRACE_MSG(" var '" << gname <<
                                      "' already clean at step-index " << t);
                    }
                }
                
            } // vars to swap.
        } // exchange sequence.

        TRACE_MSG(num_recv_reqs << " MPI receive request(s) issued");
        TRACE_MSG(num_send_reqs << " MPI send request(s) issued");
        auto mpi_call_time = halo_time.stop();
        TRACE_MSG("secs spent in MPI waits: " << make_num_str(wait_delta));
        TRACE_MSG("secs spent in this call: " << make_num_str(mpi_call_time));
        #endif
    }

    // Call MPI_Test() on all unfinished requests to advance MPI progress.
    void StencilContext::adv_halo_exchange() {

        #if defined(USE_MPI)
        STATE_VARS(this);
        if (!actl_opts->do_halo_exchange || env->num_ranks < 2)
            return;

        halo_test_time.start();
        TRACE_MSG("entering");

        // Loop thru MPI data.
        int num_tests = 0;
        for (auto& mdi : mpi_data) {
            auto& gname = mdi.first;
            auto& var_mpi_data = mdi.second;
            auto* var_recv_reqs = var_mpi_data.recv_reqs.data();
            auto* var_send_reqs = var_mpi_data.send_reqs.data();
            auto* var_recv_stats = var_mpi_data.recv_stats.data();
            auto* var_send_stats = var_mpi_data.send_stats.data();

            int flag;

            #if 1
            auto asize = max(var_mpi_data.recv_reqs.size(), var_mpi_data.send_reqs.size());
            int indices[asize];
            MPI_Status stats[asize];
            auto n = MPI_Testsome(int(var_mpi_data.recv_reqs.size()), var_recv_reqs, &flag, indices, stats);
            for (int i = 0; i < n; i++) {
                int loc = indices[i]; // Location of completed recv.
                var_recv_stats[loc] = stats[i]; // Update correct stat.
                assert(var_recv_reqs[loc] == MPI_REQUEST_NULL);
            }
            n = MPI_Testsome(int(var_mpi_data.send_reqs.size()), var_send_reqs, &flag, indices, stats);
            for (int i = 0; i < n; i++) {
                int loc = indices[i]; // Location of completed send.
                var_send_stats[loc] = stats[i]; // Update correct stat.
                assert(var_send_reqs[loc] == MPI_REQUEST_NULL);
            }

            #else
            for (size_t i = 0; i < var_mpi_data.recv_reqs.size(); i++) {
                auto& r = var_recv_reqs[i];
                if (r != MPI_REQUEST_NULL) {
                    //TRACE_MSG(gname << " recv test &MPI_Request = " << &r);
                    MPI_Test(&r, &flag, &var_recv_stats[i]);
                    num_tests++;
                    if (flag)
                        r = MPI_REQUEST_NULL;
                }
            }
            for (size_t i = 0; i < var_mpi_data.send_reqs.size(); i++) {
                auto& r = var_send_reqs[i];
                if (r != MPI_REQUEST_NULL) {
                    //TRACE_MSG(gname << " send test &MPI_Request = " << &r);
                    MPI_Test(&r, &flag, &var_send_stats[i]);
                    num_tests++;
                    if (flag)
                        r = MPI_REQUEST_NULL;
                }
            }
            #endif
        }
        auto ttime = halo_test_time.stop();
        TRACE_MSG("secs spent in " << num_tests <<
                  " MPI test(s): " << make_num_str(ttime));
        #endif
    }
    
} // namespace yask.
