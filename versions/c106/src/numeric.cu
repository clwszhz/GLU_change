#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include "numeric.h"
#include <cmath>

using namespace std;

struct DestinationGroupState {
    unsigned *lower = nullptr, *upper = nullptr, *pointer = nullptr, *target = nullptr;
    ~DestinationGroupState() {
        if (lower) cudaFree(lower); if (upper) cudaFree(upper);
        if (pointer) cudaFree(pointer); if (target) cudaFree(target);
    }
};

__global__ void destination_source_normalize(REAL *values, const unsigned *column_ptr,
        const unsigned *diagonal, const int *columns, unsigned first, unsigned end) {
    const unsigned position = first + (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    if (position >= end) return;
    const unsigned lane = threadIdx.x % warpSize, col = columns[position];
    const REAL pivot = values[diagonal[col]];
    for (unsigned p = diagonal[col] + 1 + lane; p < column_ptr[col + 1]; p += warpSize)
        values[p] /= pivot;
}

__global__ void destination_group_reduce(REAL *values, const unsigned *pointer,
        const unsigned *target, const unsigned *lower, const unsigned *upper,
        unsigned first, unsigned end) {
    const unsigned group = first + blockIdx.x * blockDim.x + threadIdx.x;
    if (group >= end) return;
    const unsigned position = target[group];
    REAL value = values[position];
    for (unsigned p = pointer[group]; p < pointer[group + 1]; ++p)
        value -= values[lower[p]] * values[upper[p]];
    values[position] = value;
}

struct DenseTailState {
    unsigned *positions = nullptr;
    REAL *values = nullptr;
    int *info = nullptr;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t exec = nullptr;
    ~DenseTailState() {
        if (exec) cudaGraphExecDestroy(exec);
        if (graph) cudaGraphDestroy(graph);
        if (positions) cudaFree(positions);
        if (values) cudaFree(values);
        if (info) cudaFree(info);
    }
};

__global__ void dense_tail_pack(const REAL *sparse, const unsigned *positions,
                                REAL *dense, unsigned count) {
    const unsigned p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p < count) dense[p] = positions[p] == ~0u ? REAL(0) : sparse[positions[p]];
}

__global__ void dense_tail_unpack(REAL *sparse, const unsigned *positions,
                                  const REAL *dense, unsigned count) {
    const unsigned p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p < count && positions[p] != ~0u) sparse[positions[p]] = dense[p];
}



__global__ void dense_panel_factor(REAL *a, unsigned n, unsigned first,
                                   unsigned end, int *info) {
    __shared__ REAL pivot;
    if (*info) return;
    for (unsigned k = first; k < end; ++k) {
        if (threadIdx.x == 0) {
            pivot = a[k + size_t(n) * k];
            if (pivot == REAL(0)) atomicCAS(info, 0, int(k + 1));
        }
        __syncthreads();
        if (*info) return;
        for (unsigned row = k + 1 + threadIdx.x; row < n; row += blockDim.x)
            a[row + size_t(n) * k] /= pivot;
        __syncthreads();
        for (unsigned row = k + 1 + threadIdx.x; row < n; row += blockDim.x) {
            const REAL l = a[row + size_t(n) * k];
            for (unsigned col = k + 1; col < end; ++col)
                a[row + size_t(n) * col] -= l * a[k + size_t(n) * col];
        }
        __syncthreads();
    }
}

// One physical warp owns one future U column, including all shuffle lanes.
__global__ void dense_panel_upper(REAL *a, unsigned n, unsigned first,
                                  unsigned end, const int *info) {
    if (*info) return;
    const unsigned lane = threadIdx.x;
    const unsigned col = end + blockIdx.x;
    const unsigned size = end - first;
    REAL value = lane < size ? a[first + lane + size_t(n) * col] : REAL(0);
    for (unsigned k = 0; k < size; ++k) {
        const REAL u = __shfl_sync(~0ULL, value, k, warpSize);
        if (lane > k && lane < size)
            value -= a[first + lane + size_t(n) * (first + k)] * u;
    }
    if (lane < size) a[first + lane + size_t(n) * col] = value;
}

__global__ void dense_panel_update(REAL *a, unsigned n, unsigned first,
                                   unsigned end, const int *info) {
    if (*info) return;
    __shared__ REAL lower[16][16], upper[16][16];
    const unsigned x = threadIdx.x, y = threadIdx.y;
    const unsigned row = end + blockIdx.x * 16 + x;
    const unsigned col = end + blockIdx.y * 16 + y;
    const unsigned size = end - first;
    lower[y][x] = row < n && y < size ? a[row + size_t(n) * (first + y)] : REAL(0);
    upper[y][x] = col < n && x < size ? a[first + x + size_t(n) * col] : REAL(0);
    __syncthreads();
    if (row < n && col < n) {
        REAL value = a[row + size_t(n) * col];
        for (unsigned k = 0; k < size; ++k) value -= lower[k][x] * upper[y][k];
        a[row + size_t(n) * col] = value;
    }
}

__global__ void initialize_sparse_values(REAL *destination, const unsigned *positions,
                                         const REAL *values, unsigned count) {
    const unsigned p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p < count) destination[positions[p]] = values[p];
}

__device__ __forceinline__ REAL scratch_value(const REAL *values, const unsigned *tags,
                                              size_t index, unsigned column) {
    return tags[index] == column + 1u ? values[index] : REAL(0);
}


__device__ __forceinline__ void RL_body(
        const unsigned* __restrict__ sym_c_ptr_dev,
        const unsigned* __restrict__ sym_r_idx_dev,
        REAL* __restrict__ val_dev,
        const unsigned* __restrict__ l_col_ptr_dev,
        const unsigned* __restrict__ csr_r_ptr_dev,
        const unsigned* __restrict__ csr_c_idx_dev,
        const unsigned* __restrict__ csr_diag_ptr_dev,
        const unsigned* __restrict__ csr_pivot_pos_dev,
        const int* __restrict__ level_idx_dev,
        unsigned* __restrict__ scratch_tag,
        REAL* __restrict__ tmpMem,
        const unsigned n,
        const int levelHead,
        const int inLevPos, const int scratch_slot)
{
    const int tid = threadIdx.x;
    const int bid = scratch_slot;
    const int wid = threadIdx.x / warpSize;

    const unsigned currentCol = level_idx_dev[levelHead+inLevPos+bid];
    const unsigned currentLColSize = sym_c_ptr_dev[currentCol + 1] - l_col_ptr_dev[currentCol] - 1;
    const unsigned currentLPos = l_col_ptr_dev[currentCol] + tid + 1;


    //update current col

    int offset = 0;
    while (currentLColSize > offset)
    {
        if (tid + offset < currentLColSize)
        {
            unsigned ridx = sym_r_idx_dev[currentLPos + offset];

            val_dev[currentLPos + offset] /= val_dev[l_col_ptr_dev[currentCol]];
            tmpMem[bid*n + ridx]= val_dev[currentLPos + offset];
            scratch_tag[bid*n + ridx] = currentCol + 1u;
        }
        offset += blockDim.x;
    }
    __syncthreads();

    // Each warp owns one destination column. Its leader finds the sorted
    // pivot, then broadcasts it explicitly before lanes update lower rows.
    const unsigned lane = threadIdx.x % warpSize;
    const unsigned long long native_mask = ~0ULL;
    const unsigned row_end = csr_r_ptr_dev[currentCol + 1];
    for (unsigned entry = csr_diag_ptr_dev[currentCol] + 1 + wid;
         entry < row_end; entry += blockDim.x / warpSize) {
        const unsigned subCol = csr_c_idx_dev[entry];
        const unsigned begin = sym_c_ptr_dev[subCol];
        const unsigned end = sym_c_ptr_dev[subCol + 1];
        // A destination belongs to one whole physical warp. Full-width
        // ballot and shuffle preserve all participants on warp32 and warp64.
        if (end - begin <= 64 && csr_pivot_pos_dev[entry] == ~0u) {
            REAL short_coefficient = 0;
            for (unsigned tile = begin; tile < end; tile += warpSize) {
                const unsigned pos = tile + lane;
                const bool active = pos < end;
                const unsigned row = active ? sym_r_idx_dev[pos] : 0;
                const bool is_pivot = active && row == currentCol;
                const REAL pivot_value = is_pivot ? val_dev[pos] : REAL(0);
                const unsigned long long pivot_mask = __ballot_sync(native_mask, is_pivot);
                if (pivot_mask != 0) {
                    const int pivot_lane = 63 - __clzll(pivot_mask);
                    short_coefficient = __shfl_sync(native_mask, pivot_value, pivot_lane, warpSize);
                }
                if (active && row > currentCol)
                    atomicAdd(&val_dev[pos], -scratch_value(tmpMem, scratch_tag, size_t(bid) * n + row, currentCol) * short_coefficient);
            }
            continue;
        }
        unsigned first_lower = 0;
        REAL coefficient = 0;
        if (lane == 0) {
            const unsigned pivot = csr_pivot_pos_dev[entry];
            if (pivot != ~0u) {
                first_lower = pivot + 1;
                coefficient = val_dev[pivot];
            } else {
                // Retain the legacy upper-bound semantics for duplicates.
                unsigned lo = begin, hi = end;
                while (lo < hi) {
                    const unsigned mid = lo + (hi - lo) / 2;
                    if (sym_r_idx_dev[mid] <= currentCol) lo = mid + 1;
                    else hi = mid;
                }
                first_lower = lo;
                if (lo > begin && sym_r_idx_dev[lo - 1] == currentCol)
                    coefficient = val_dev[lo - 1];
            }
        }
        first_lower = __shfl_sync(native_mask, first_lower, 0, warpSize);
        coefficient = __shfl_sync(native_mask, coefficient, 0, warpSize);
        // In a very long destination, most dense scratch entries may be
        // structural zeros. Search only the source pattern in that case;
        // retain contiguous destination traversal for denser intersections.
        // This predicate is uniform within the native warp.
        if (currentLColSize > 0 && (end - first_lower) / 32u > currentLColSize) {
            const unsigned source_begin = l_col_ptr_dev[currentCol] + 1;
            const unsigned source_end = sym_c_ptr_dev[currentCol + 1];
            for (unsigned src = source_begin + lane; src < source_end; src += warpSize) {
                const unsigned row = sym_r_idx_dev[src];
                // The legacy symbolic fallback can retain duplicates. The
                // dense algorithm reads one scratch value for each row;
                // visit that row once and update every equal target entry.
                if (src > source_begin && sym_r_idx_dev[src - 1] == row)
                    continue;
                unsigned lo = first_lower;
                unsigned hi = end;
                while (lo < hi) {
                    const unsigned mid = lo + (hi - lo) / 2;
                    if (sym_r_idx_dev[mid] < row) lo = mid + 1;
                    else hi = mid;
                }
                while (lo < end && sym_r_idx_dev[lo] == row) {
                    atomicAdd(&val_dev[lo], -scratch_value(tmpMem, scratch_tag, size_t(bid) * n + row, currentCol) * coefficient);
                    ++lo;
                }
            }
        } else {
            for (unsigned pos = first_lower + lane; pos < end; pos += warpSize) {
                const unsigned row = sym_r_idx_dev[pos];
                atomicAdd(&val_dev[pos], -scratch_value(tmpMem, scratch_tag, size_t(bid) * n + row, currentCol) * coefficient);
            }
        }
    }

}

__global__ void RL(
        const unsigned* __restrict__ sym_c_ptr_dev,
        const unsigned* __restrict__ sym_r_idx_dev,
        REAL* __restrict__ val_dev,
        const unsigned* __restrict__ l_col_ptr_dev,
        const unsigned* __restrict__ csr_r_ptr_dev,
        const unsigned* __restrict__ csr_c_idx_dev,
        const unsigned* __restrict__ csr_diag_ptr_dev,
        const unsigned* __restrict__ csr_pivot_pos_dev,
        const int* __restrict__ level_idx_dev,
        unsigned* __restrict__ scratch_tag,
        REAL* __restrict__ tmpMem,
        const unsigned n,
        const int levelHead,
        const int inLevPos)
{
    RL_body(sym_c_ptr_dev, sym_r_idx_dev, val_dev, l_col_ptr_dev, csr_r_ptr_dev, csr_c_idx_dev, csr_diag_ptr_dev, csr_pivot_pos_dev, level_idx_dev, scratch_tag, tmpMem, n, levelHead, inLevPos, blockIdx.x);
}

// One CTA owns the complete bounded chain. The barrier publishes global
// factor/scratch writes before any source in the next level reads them.
__global__ void RL_narrow_chain(
        const unsigned* sym_c_ptr_dev, const unsigned* sym_r_idx_dev, REAL* val_dev,
        const unsigned* l_col_ptr_dev, const unsigned* csr_r_ptr_dev,
        const unsigned* csr_c_idx_dev, const unsigned* csr_diag_ptr_dev,
        const unsigned* csr_pivot_pos_dev,
        const int* level_idx_dev, unsigned* scratch_tag, REAL* tmpMem, unsigned n,
        const int* level_ptr_dev, unsigned first_level, unsigned end_level) {
    for (unsigned lv = first_level; lv < end_level; ++lv) {
        for (int pos = level_ptr_dev[lv]; pos < level_ptr_dev[lv + 1]; ++pos) {
            RL_body(sym_c_ptr_dev, sym_r_idx_dev, val_dev, l_col_ptr_dev, csr_r_ptr_dev, csr_c_idx_dev, csr_diag_ptr_dev, csr_pivot_pos_dev, level_idx_dev, scratch_tag, tmpMem, n, pos, 0, 0);
            __syncthreads();
        }
    }
}

__global__ void RL_perturb(
        const unsigned* __restrict__ sym_c_ptr_dev,
        const unsigned* __restrict__ sym_r_idx_dev,
        REAL* __restrict__ val_dev,
        const unsigned* __restrict__ l_col_ptr_dev,
        const unsigned* __restrict__ csr_r_ptr_dev,
        const unsigned* __restrict__ csr_c_idx_dev,
        const unsigned* __restrict__ csr_diag_ptr_dev,
        const unsigned* __restrict__ csr_pivot_pos_dev,
        const int* __restrict__ level_idx_dev,
        unsigned* __restrict__ scratch_tag,
        REAL* __restrict__ tmpMem,
        const unsigned n,
        const int levelHead,
        const int inLevPos,
        const REAL pert)
{
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int wid = threadIdx.x / warpSize;

    const unsigned currentCol = level_idx_dev[levelHead+inLevPos+bid];
    const unsigned currentLColSize = sym_c_ptr_dev[currentCol + 1] - l_col_ptr_dev[currentCol] - 1;
    const unsigned currentLPos = l_col_ptr_dev[currentCol] + tid + 1;


    //update current col

    int offset = 0;
    while (currentLColSize > offset)
    {
        if (tid + offset < currentLColSize)
        {
            unsigned ridx = sym_r_idx_dev[currentLPos + offset];

            if (fabs(val_dev[l_col_ptr_dev[currentCol]]) < pert)
                val_dev[l_col_ptr_dev[currentCol]] = pert;

            val_dev[currentLPos + offset] /= val_dev[l_col_ptr_dev[currentCol]];
            tmpMem[bid*n + ridx]= val_dev[currentLPos + offset];
            scratch_tag[bid*n + ridx] = currentCol + 1u;
        }
        offset += blockDim.x;
    }
    __syncthreads();

    // Each warp owns one destination column. Its leader finds the sorted
    // pivot, then broadcasts it explicitly before lanes update lower rows.
    const unsigned lane = threadIdx.x % warpSize;
    const unsigned long long native_mask = ~0ULL;
    const unsigned row_end = csr_r_ptr_dev[currentCol + 1];
    for (unsigned entry = csr_diag_ptr_dev[currentCol] + 1 + wid;
         entry < row_end; entry += blockDim.x / warpSize) {
        const unsigned subCol = csr_c_idx_dev[entry];
        const unsigned begin = sym_c_ptr_dev[subCol];
        const unsigned end = sym_c_ptr_dev[subCol + 1];
        // A destination belongs to one whole physical warp. Full-width
        // ballot and shuffle preserve all participants on warp32 and warp64.
        if (end - begin <= 64 && csr_pivot_pos_dev[entry] == ~0u) {
            REAL short_coefficient = 0;
            for (unsigned tile = begin; tile < end; tile += warpSize) {
                const unsigned pos = tile + lane;
                const bool active = pos < end;
                const unsigned row = active ? sym_r_idx_dev[pos] : 0;
                const bool is_pivot = active && row == currentCol;
                const REAL pivot_value = is_pivot ? val_dev[pos] : REAL(0);
                const unsigned long long pivot_mask = __ballot_sync(native_mask, is_pivot);
                if (pivot_mask != 0) {
                    const int pivot_lane = 63 - __clzll(pivot_mask);
                    short_coefficient = __shfl_sync(native_mask, pivot_value, pivot_lane, warpSize);
                }
                if (active && row > currentCol)
                    atomicAdd(&val_dev[pos], -scratch_value(tmpMem, scratch_tag, size_t(bid) * n + row, currentCol) * short_coefficient);
            }
            continue;
        }
        unsigned first_lower = 0;
        REAL coefficient = 0;
        if (lane == 0) {
            const unsigned pivot = csr_pivot_pos_dev[entry];
            if (pivot != ~0u) {
                first_lower = pivot + 1;
                coefficient = val_dev[pivot];
            } else {
                // Retain the legacy upper-bound semantics for duplicates.
                unsigned lo = begin, hi = end;
                while (lo < hi) {
                    const unsigned mid = lo + (hi - lo) / 2;
                    if (sym_r_idx_dev[mid] <= currentCol) lo = mid + 1;
                    else hi = mid;
                }
                first_lower = lo;
                if (lo > begin && sym_r_idx_dev[lo - 1] == currentCol)
                    coefficient = val_dev[lo - 1];
            }
        }
        first_lower = __shfl_sync(native_mask, first_lower, 0, warpSize);
        coefficient = __shfl_sync(native_mask, coefficient, 0, warpSize);
        // In a very long destination, most dense scratch entries may be
        // structural zeros. Search only the source pattern in that case;
        // retain contiguous destination traversal for denser intersections.
        // This predicate is uniform within the native warp.
        if (currentLColSize > 0 && (end - first_lower) / 32u > currentLColSize) {
            const unsigned source_begin = l_col_ptr_dev[currentCol] + 1;
            const unsigned source_end = sym_c_ptr_dev[currentCol + 1];
            for (unsigned src = source_begin + lane; src < source_end; src += warpSize) {
                const unsigned row = sym_r_idx_dev[src];
                // The legacy symbolic fallback can retain duplicates. The
                // dense algorithm reads one scratch value for each row;
                // visit that row once and update every equal target entry.
                if (src > source_begin && sym_r_idx_dev[src - 1] == row)
                    continue;
                unsigned lo = first_lower;
                unsigned hi = end;
                while (lo < hi) {
                    const unsigned mid = lo + (hi - lo) / 2;
                    if (sym_r_idx_dev[mid] < row) lo = mid + 1;
                    else hi = mid;
                }
                while (lo < end && sym_r_idx_dev[lo] == row) {
                    atomicAdd(&val_dev[lo], -scratch_value(tmpMem, scratch_tag, size_t(bid) * n + row, currentCol) * coefficient);
                    ++lo;
                }
            }
        } else {
            for (unsigned pos = first_lower + lane; pos < end; pos += warpSize) {
                const unsigned row = sym_r_idx_dev[pos];
                atomicAdd(&val_dev[pos], -scratch_value(tmpMem, scratch_tag, size_t(bid) * n + row, currentCol) * coefficient);
            }
        }
    }

}

// Fully cached update tasks read val directly, so these sources need only
// one normalization and no dense scratch/tag scatter. Each native warp owns
// an independent source; completion precedes updates on the same stream.
__global__ void RL_cached_source_prepare(const unsigned *cp, REAL *val,
        const unsigned *diag, const int *level_idx, const unsigned *source_queue,
        unsigned queue_first, unsigned queue_end) {
    const unsigned owner = blockIdx.x * (blockDim.x / warpSize) + threadIdx.x / warpSize;
    const unsigned position = queue_first + owner;
    if (position >= queue_end) return;
    const unsigned col = level_idx[source_queue[position]];
    const unsigned lane = threadIdx.x % warpSize;
    const REAL pivot = val[diag[col]];
    for (unsigned pos = diag[col] + 1 + lane; pos < cp[col + 1]; pos += warpSize)
        val[pos] /= pivot;
}

// Values and their source tags are published before the ordered update.
__global__ void RL_batch_prepare(
        const unsigned *cp, const unsigned *ri, REAL *val,
        const unsigned *diag, const int *level_idx, unsigned *tags, REAL *scratch,
        unsigned n, unsigned source_head, bool perturb, REAL pert,
        const unsigned *source_queue, unsigned queue_first) {
    const unsigned slot = source_queue ? source_queue[queue_first + blockIdx.x] - source_head : blockIdx.x;
    const unsigned col = level_idx[source_head + slot];
    if (perturb && threadIdx.x == 0 && fabs(val[diag[col]]) < pert)
        val[diag[col]] = pert;
    __syncthreads();
    for (unsigned pos = diag[col] + 1 + threadIdx.x; pos < cp[col + 1]; pos += blockDim.x) {
        val[pos] /= val[diag[col]];
        const size_t index = size_t(slot) * n + ri[pos];
        scratch[index] = val[pos];
        tags[index] = col + 1u;
    }
}

__global__ void RL_update_short_tasks(REAL *val,
        const unsigned *pivot, const unsigned *task_entry, const unsigned *task_index,
        const unsigned *pair_begin, const unsigned *pair_end,
        const unsigned *pair_source, const unsigned *pair_target,
        unsigned head, unsigned end) {
    const unsigned logical = head + blockIdx.x * (blockDim.x / 16) + threadIdx.x / 16;
    if (logical >= end) return; // no warp collectives in this kernel
    const unsigned task = task_index[logical];
    const unsigned p = pair_begin[task] + threadIdx.x % 16;
    if (p < pair_end[task]) {
        const REAL coefficient = val[pivot[task_entry[task]]];
        atomicAdd(&val[pair_target[p]], -val[pair_source[p]] * coefficient);
    }
}

__global__ void RL_update_tasks(
        const unsigned* __restrict__ sym_c_ptr_dev,
        const unsigned* __restrict__ sym_r_idx_dev,
        REAL* __restrict__ val_dev,
        const unsigned* __restrict__ l_col_ptr_dev,
        const unsigned* __restrict__ csr_c_idx_dev,
        const int* __restrict__ level_idx_dev,
        const unsigned* __restrict__ csr_pivot_pos_dev,
        const unsigned* __restrict__ task_entry,
        const unsigned* __restrict__ task_source,
        const unsigned* __restrict__ task_begin,
        const unsigned* __restrict__ task_end_pos,
        const unsigned* __restrict__ pair_begin, const unsigned* __restrict__ pair_end,
        const unsigned* __restrict__ pair_source, const unsigned* __restrict__ pair_target,
        unsigned* __restrict__ scratch_tag,
        REAL* __restrict__ tmpMem,
        unsigned n, unsigned source_head, unsigned task_head, unsigned task_end,
        bool skip_short_tasks) {
    const unsigned tasks_per_block = blockDim.x / warpSize;
    const unsigned task = task_head + blockIdx.x * tasks_per_block + threadIdx.x / warpSize;
    if (task >= task_end) return; // whole native warp exits together
    if (skip_short_tasks && pair_begin[task] != ~0u &&
        pair_end[task] - pair_begin[task] <= 16) return;
    const unsigned source_pos = task_source[task];
    const unsigned bid = source_pos - source_head;
    const unsigned currentCol = level_idx_dev[source_pos];
    const unsigned currentLColSize = sym_c_ptr_dev[currentCol + 1] - l_col_ptr_dev[currentCol] - 1;
    // Each warp owns one destination column. Its leader finds the sorted
    // pivot, then broadcasts it explicitly before lanes update lower rows.
    const unsigned lane = threadIdx.x % warpSize;
    const unsigned long long native_mask = ~0ULL;
    for (unsigned once = 0; once < 1; ++once) {
        const unsigned entry = task_entry[task];
        const unsigned subCol = csr_c_idx_dev[entry];
        const unsigned begin = sym_c_ptr_dev[subCol];
        const unsigned end = sym_c_ptr_dev[subCol + 1];
        if (pair_begin[task] != ~0u) {
            const REAL coefficient = val_dev[csr_pivot_pos_dev[entry]];
            for (unsigned p = pair_begin[task] + lane; p < pair_end[task]; p += warpSize)
                atomicAdd(&val_dev[pair_target[p]], -val_dev[pair_source[p]] * coefficient);
            continue;
        }
        if (task_begin[task] != ~0u) {
            const REAL coefficient = val_dev[csr_pivot_pos_dev[entry]];
            for (unsigned pos = task_begin[task] + lane; pos < task_end_pos[task]; pos += warpSize) {
                const unsigned row = sym_r_idx_dev[pos];
                atomicAdd(&val_dev[pos], -scratch_value(tmpMem, scratch_tag, size_t(bid) * n + row, currentCol) * coefficient);
            }
            continue;
        }
        // A destination belongs to one whole physical warp. Full-width
        // ballot and shuffle preserve all participants on warp32 and warp64.
        if (end - begin <= 64) {
            REAL short_coefficient = 0;
            for (unsigned tile = begin; tile < end; tile += warpSize) {
                const unsigned pos = tile + lane;
                const bool active = pos < end;
                const unsigned row = active ? sym_r_idx_dev[pos] : 0;
                const bool is_pivot = active && row == currentCol;
                const REAL pivot_value = is_pivot ? val_dev[pos] : REAL(0);
                const unsigned long long pivot_mask = __ballot_sync(native_mask, is_pivot);
                if (pivot_mask != 0) {
                    const int pivot_lane = 63 - __clzll(pivot_mask);
                    short_coefficient = __shfl_sync(native_mask, pivot_value, pivot_lane, warpSize);
                }
                if (active && row > currentCol)
                    atomicAdd(&val_dev[pos], -scratch_value(tmpMem, scratch_tag, size_t(bid) * n + row, currentCol) * short_coefficient);
            }
            continue;
        }
        unsigned first_lower = 0;
        REAL coefficient = 0;
        if (lane == 0) {
            const unsigned pivot = csr_pivot_pos_dev[entry];
            if (pivot != ~0u) {
                first_lower = pivot + 1;
                coefficient = val_dev[pivot];
            } else {
                // Duplicate-index columns retain the original upper-bound rule.
                unsigned lo = begin, hi = end;
                while (lo < hi) {
                    const unsigned mid = lo + (hi - lo) / 2;
                    if (sym_r_idx_dev[mid] <= currentCol) lo = mid + 1;
                    else hi = mid;
                }
                first_lower = lo;
                if (lo > begin && sym_r_idx_dev[lo - 1] == currentCol)
                    coefficient = val_dev[lo - 1];
            }
        }
        first_lower = __shfl_sync(native_mask, first_lower, 0, warpSize);
        coefficient = __shfl_sync(native_mask, coefficient, 0, warpSize);
        // In a very long destination, most dense scratch entries may be
        // structural zeros. Search only the source pattern in that case;
        // retain contiguous destination traversal for denser intersections.
        // This predicate is uniform within the native warp.
        if (currentLColSize > 0 && (end - first_lower) / 32u > currentLColSize) {
            const unsigned source_begin = l_col_ptr_dev[currentCol] + 1;
            const unsigned source_end = sym_c_ptr_dev[currentCol + 1];
            for (unsigned src = source_begin + lane; src < source_end; src += warpSize) {
                const unsigned row = sym_r_idx_dev[src];
                // The legacy symbolic fallback can retain duplicates. The
                // dense algorithm reads one scratch value for each row;
                // visit that row once and update every equal target entry.
                if (src > source_begin && sym_r_idx_dev[src - 1] == row)
                    continue;
                unsigned lo = first_lower;
                unsigned hi = end;
                while (lo < hi) {
                    const unsigned mid = lo + (hi - lo) / 2;
                    if (sym_r_idx_dev[mid] < row) lo = mid + 1;
                    else hi = mid;
                }
                while (lo < end && sym_r_idx_dev[lo] == row) {
                    atomicAdd(&val_dev[lo], -scratch_value(tmpMem, scratch_tag, size_t(bid) * n + row, currentCol) * coefficient);
                    ++lo;
                }
            }
        } else {
            for (unsigned pos = first_lower + lane; pos < end; pos += warpSize) {
                const unsigned row = sym_r_idx_dev[pos];
                atomicAdd(&val_dev[pos], -scratch_value(tmpMem, scratch_tag, size_t(bid) * n + row, currentCol) * coefficient);
            }
        }
    }

}

__global__ void RL_onecol_factorizeCurrentCol(
        const unsigned* __restrict__ sym_c_ptr_dev,
        const unsigned* __restrict__ sym_r_idx_dev,
        REAL* __restrict__ val_dev,
        const unsigned* __restrict__ l_col_ptr_dev,
        const unsigned currentCol,
        unsigned* __restrict__ scratch_tag,
        REAL* __restrict__ tmpMem,
        const int stream,
        const unsigned n)
{
    const int tid = threadIdx.x;

    const unsigned currentLColSize = sym_c_ptr_dev[currentCol + 1] - l_col_ptr_dev[currentCol] - 1;
    const unsigned currentLPos = l_col_ptr_dev[currentCol] + tid + 1;

    //update current col

    int offset = 0;
    while (currentLColSize > offset)
    {
        if (tid + offset < currentLColSize)
        {
            unsigned ridx = sym_r_idx_dev[currentLPos + offset];

            val_dev[currentLPos + offset] /= val_dev[l_col_ptr_dev[currentCol]];
            tmpMem[stream * n + ridx]= val_dev[currentLPos + offset];
            scratch_tag[stream * n + ridx] = currentCol + 1u;
        }
        offset += blockDim.x;
    }
}

__global__ void RL_onecol_factorizeCurrentCol_perturb(
        const unsigned* __restrict__ sym_c_ptr_dev,
        const unsigned* __restrict__ sym_r_idx_dev,
        REAL* __restrict__ val_dev,
        const unsigned* __restrict__ l_col_ptr_dev,
        const unsigned currentCol,
        unsigned* __restrict__ scratch_tag,
        REAL* __restrict__ tmpMem,
        const int stream,
        const unsigned n,
        const REAL pert)
{
    const int tid = threadIdx.x;

    const unsigned currentLColSize = sym_c_ptr_dev[currentCol + 1] - l_col_ptr_dev[currentCol] - 1;
    const unsigned currentLPos = l_col_ptr_dev[currentCol] + tid + 1;

    //update current col

    int offset = 0;
    while (currentLColSize > offset)
    {
        if (tid + offset < currentLColSize)
        {
            unsigned ridx = sym_r_idx_dev[currentLPos + offset];

            if (fabs(val_dev[l_col_ptr_dev[currentCol]]) < pert)
                val_dev[l_col_ptr_dev[currentCol]] = pert;

            val_dev[currentLPos + offset] /= val_dev[l_col_ptr_dev[currentCol]];
            tmpMem[stream * n + ridx]= val_dev[currentLPos + offset];
            scratch_tag[stream * n + ridx] = currentCol + 1u;
        }
        offset += blockDim.x;
    }
}

__global__ void RL_onecol_updateSubmat(
        const unsigned* __restrict__ sym_c_ptr_dev,
        const unsigned* __restrict__ sym_r_idx_dev,
        REAL* __restrict__ val_dev,
        const unsigned* __restrict__ csr_c_idx_dev,
        const unsigned* __restrict__ csr_diag_ptr_dev,
        const unsigned* __restrict__ csr_pivot_pos_dev,
        const unsigned currentCol,
        unsigned* __restrict__ scratch_tag,
        REAL* __restrict__ tmpMem,
        const int stream,
        const unsigned n)
{
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    __shared__ REAL s;

    //broadcast to submatrix
    const unsigned subColPos = csr_diag_ptr_dev[currentCol] + bid + 1;
    unsigned subCol;
    unsigned subColElem = 0;

    int offset = 0;
    subCol = csr_c_idx_dev[subColPos];
    const unsigned pivot = csr_pivot_pos_dev[subColPos];
    if (pivot != ~0u) {
        const REAL coefficient = val_dev[pivot];
        for (unsigned pos = pivot + 1 + tid; pos < sym_c_ptr_dev[subCol + 1]; pos += blockDim.x) {
            const unsigned row = sym_r_idx_dev[pos];
            atomicAdd(&val_dev[pos], -scratch_value(tmpMem, scratch_tag, size_t(stream) * n + row, currentCol) * coefficient);
        }
        return;
    }
    // Duplicate-index columns keep the original scan and barrier semantics.

    if (tid == 0)
        s = 0;
    __syncthreads();
    while(offset < sym_c_ptr_dev[subCol + 1] - sym_c_ptr_dev[subCol])
    {
        bool active = (tid + offset < sym_c_ptr_dev[subCol + 1] - sym_c_ptr_dev[subCol]);
        unsigned ridx = 0;
        if (active)
        {
            subColElem = sym_c_ptr_dev[subCol] + tid + offset;
            ridx = sym_r_idx_dev[subColElem];

            if (ridx == currentCol)
            {
                s = val_dev[subColElem];
            }
        }
        // Every thread must reach both barriers; gating __syncthreads by "active"
        // can deadlock when sub-column nnz is smaller than blockDim.x.
        __syncthreads();
        if (active && ridx > currentCol)
        {
            atomicAdd(&val_dev[subColElem], -scratch_value(tmpMem, scratch_tag, stream * n + ridx, currentCol) * s);
        }
        __syncthreads();
        offset += blockDim.x;
    }
}

void LUonDevice(Symbolic_Matrix &A_sym, ostream &out, ostream &err, bool PERTURB, PhaseReport *report)
{
    if (A_sym.n == 0 || A_sym.nnz == 0) {
        err << "Matrix is empty; skipping GPU factorization." << endl;
        return;
    }

    auto cudaCheck = [&](cudaError_t status, const char *op) -> bool {
        if (status != cudaSuccess) {
            err << op << " failed: " << cudaGetErrorString(status) << endl;
            return false;
        }
        return true;
    };

    unsigned n = A_sym.n;
    unsigned nnz = A_sym.nnz;
    unsigned num_lev = A_sym.num_lev;

    const auto stamp = [&]() { return report ? report->now() : 0.0; };
    const auto record = [&](const char *name, double start, bool reusable) { if (report) report->add(name, start, reusable); };
    double phase_start = stamp();
    int deviceCount = 0;
    if (!cudaCheck(cudaGetDeviceCount(&deviceCount), "cudaGetDeviceCount"))
        return;
    if (deviceCount <= 0) {
        err << "No CUDA-capable GPU detected." << endl;
        return;
    }

    int dev = 0;
    cudaDeviceProp deviceProp;
    if (!cudaCheck(cudaGetDeviceProperties(&deviceProp, dev), "cudaGetDeviceProperties"))
        return;
    if (!cudaCheck(cudaSetDevice(dev), "cudaSetDevice"))
        return;
    out << "Device " << dev << ": " << deviceProp.name << " has been selected." << endl;

    // Force lazy context creation into the initialization interval.
    if (!cudaCheck(cudaFree(0), "initialize context") ||
        !cudaCheck(cudaDeviceSynchronize(), "initialize synchronize")) return;
    record("gpu_initialization", phase_start, true);
    cudaEvent_t start = nullptr, stop = nullptr;
    unsigned *sym_c_ptr_dev = nullptr, *sym_r_idx_dev = nullptr, *l_col_ptr_dev = nullptr;
    REAL *val_dev = nullptr, *numeric_values_dev = nullptr;
    unsigned *numeric_dst_dev = nullptr;
    bool factor_host_registered = false;
    const bool device_assembly = A_sym.numeric_device_scatter;
    const size_t numeric_entries = device_assembly ? A_sym.numeric_scatter_dst.size() : 0;
    unsigned *csr_r_ptr_dev = nullptr, *csr_c_idx_dev = nullptr, *csr_diag_ptr_dev = nullptr;
    int *level_idx_dev = nullptr;
    int *level_ptr_dev = nullptr;
    unsigned *csr_pivot_pos_dev = nullptr;
    unsigned *batch_task_entry_dev = nullptr, *batch_task_source_dev = nullptr;
    unsigned *batch_task_begin_dev = nullptr, *batch_task_end_dev = nullptr;
    unsigned *update_pair_begin_dev = nullptr, *update_pair_end_dev = nullptr;
    unsigned *update_pair_source_dev = nullptr, *update_pair_target_dev = nullptr;
    unsigned *short_update_task_index_dev = nullptr;
    unsigned *cached_prepare_sources_dev = nullptr, *scratch_prepare_sources_dev = nullptr;
    REAL *tmpMem = nullptr;
    unsigned *scratch_tag_dev = nullptr;
    float time = 0.0f;
    cudaStream_t capture_stream = nullptr;
    bool capture_active = false;
    std::vector<cudaGraph_t> level_graph(num_lev, nullptr);
    std::vector<cudaGraphExec_t> level_exec(num_lev, nullptr);
    std::vector<unsigned> graph_end(num_lev, 0);
    unsigned graph_segments = 0, graph_levels = 0;
    std::shared_ptr<DenseTailState> dense_state;
    std::shared_ptr<DestinationGroupState> destination_state;
    const bool destination_owned = !PERTURB && A_sym.destination_ready;
    const unsigned dense_size = (PERTURB || destination_owned) ? 0u : A_sym.dense_columns.size();
    const auto dense_level = [&](unsigned lv) {
        return dense_size && int(lv) >= A_sym.dense_first_level && int(lv) < A_sym.dense_end_level;
    };

    constexpr int Nstreams = 16;
    cudaStream_t streams[Nstreams];
    bool stream_created[Nstreams] = {false};

    auto cleanup = [&]() {
        if (capture_active) {
            cudaGraph_t abandoned = nullptr;
            cudaStreamEndCapture(capture_stream, &abandoned);
            if (abandoned) cudaGraphDestroy(abandoned);
            capture_active = false;
        }
        dense_state.reset();
        destination_state.reset();
        for (unsigned i = 0; i < num_lev; ++i) {
            if (level_exec[i]) cudaGraphExecDestroy(level_exec[i]);
            if (level_graph[i]) cudaGraphDestroy(level_graph[i]);
        }
        if (capture_stream) cudaStreamDestroy(capture_stream);
        for (int j = 0; j < Nstreams; ++j) {
            if (stream_created[j])
                cudaStreamDestroy(streams[j]);
        }
        if (tmpMem != nullptr)
            cudaFree(tmpMem);
        if (scratch_tag_dev != nullptr) cudaFree(scratch_tag_dev);
        if (sym_c_ptr_dev != nullptr)
            cudaFree(sym_c_ptr_dev);
        if (sym_r_idx_dev != nullptr)
            cudaFree(sym_r_idx_dev);
        if (val_dev != nullptr)
            cudaFree(val_dev);
        if (numeric_values_dev) cudaFree(numeric_values_dev);
        if (numeric_dst_dev) cudaFree(numeric_dst_dev);
        if (l_col_ptr_dev != nullptr)
            cudaFree(l_col_ptr_dev);
        if (csr_c_idx_dev != nullptr)
            cudaFree(csr_c_idx_dev);
        if (csr_r_ptr_dev != nullptr)
            cudaFree(csr_r_ptr_dev);
        if (csr_diag_ptr_dev != nullptr)
            cudaFree(csr_diag_ptr_dev);
        if (batch_task_entry_dev != nullptr) cudaFree(batch_task_entry_dev);
        if (batch_task_source_dev != nullptr) cudaFree(batch_task_source_dev);
        if (batch_task_begin_dev != nullptr) cudaFree(batch_task_begin_dev);
        if (batch_task_end_dev != nullptr) cudaFree(batch_task_end_dev);
        if (update_pair_begin_dev != nullptr) cudaFree(update_pair_begin_dev);
        if (update_pair_end_dev != nullptr) cudaFree(update_pair_end_dev);
        if (update_pair_source_dev != nullptr) cudaFree(update_pair_source_dev);
        if (update_pair_target_dev != nullptr) cudaFree(update_pair_target_dev);
        if (short_update_task_index_dev != nullptr) cudaFree(short_update_task_index_dev);
        if (cached_prepare_sources_dev) cudaFree(cached_prepare_sources_dev);
        if (scratch_prepare_sources_dev) cudaFree(scratch_prepare_sources_dev);
        if (csr_pivot_pos_dev != nullptr) cudaFree(csr_pivot_pos_dev);
        if (level_idx_dev != nullptr)
            cudaFree(level_idx_dev);
        if (level_ptr_dev != nullptr) cudaFree(level_ptr_dev);
        if (start != nullptr)
            cudaEventDestroy(start);
        if (stop != nullptr)
            cudaEventDestroy(stop);
        if (factor_host_registered) {
            cudaHostUnregister(A_sym.val.data());
            factor_host_registered = false;
        }
    };

#define CUDA_RETURN_ON_ERR(call, op_name) \
    do { \
        if (!cudaCheck((call), (op_name))) { \
            cleanup(); \
            return; \
        } \
    } while (0)

    phase_start = stamp();
    CUDA_RETURN_ON_ERR(cudaEventCreate(&start), "cudaEventCreate(start)");
    CUDA_RETURN_ON_ERR(cudaEventCreate(&stop), "cudaEventCreate(stop)");
    CUDA_RETURN_ON_ERR(cudaEventRecord(start, 0), "cudaEventRecord(start)");

    // The receive storage stays at the same address for the whole factorization.
    // Register only large transfers; unregister remains in normal charged cleanup.
    if (nnz > (1u << 20)) {
        CUDA_RETURN_ON_ERR(cudaHostRegister(A_sym.val.data(), nnz * sizeof(REAL), cudaHostRegisterDefault),
            "cudaHostRegister(factor receive)");
        factor_host_registered = true;
    }

    CUDA_RETURN_ON_ERR(cudaMalloc((void**)&sym_c_ptr_dev, (n + 1) * sizeof(unsigned)), "cudaMalloc(sym_c_ptr_dev)");
    CUDA_RETURN_ON_ERR(cudaMalloc((void**)&sym_r_idx_dev, nnz * sizeof(unsigned)), "cudaMalloc(sym_r_idx_dev)");
    CUDA_RETURN_ON_ERR(cudaMalloc((void**)&val_dev, nnz * sizeof(REAL)), "cudaMalloc(val_dev)");
    if (numeric_entries) {
        CUDA_RETURN_ON_ERR(cudaMalloc((void**)&numeric_values_dev, numeric_entries * sizeof(REAL)), "cudaMalloc(numeric_values)");
        CUDA_RETURN_ON_ERR(cudaMalloc((void**)&numeric_dst_dev, numeric_entries * sizeof(unsigned)), "cudaMalloc(numeric_destinations)");
    }
    CUDA_RETURN_ON_ERR(cudaMalloc((void**)&l_col_ptr_dev, n * sizeof(unsigned)), "cudaMalloc(l_col_ptr_dev)");
    CUDA_RETURN_ON_ERR(cudaMalloc((void**)&csr_r_ptr_dev, (n + 1) * sizeof(unsigned)), "cudaMalloc(csr_r_ptr_dev)");
    CUDA_RETURN_ON_ERR(cudaMalloc((void**)&csr_c_idx_dev, nnz * sizeof(unsigned)), "cudaMalloc(csr_c_idx_dev)");
    CUDA_RETURN_ON_ERR(cudaMalloc((void**)&csr_diag_ptr_dev, n * sizeof(unsigned)), "cudaMalloc(csr_diag_ptr_dev)");
    CUDA_RETURN_ON_ERR(cudaMalloc((void**)&level_idx_dev, n * sizeof(int)), "cudaMalloc(level_idx_dev)");
    CUDA_RETURN_ON_ERR(cudaMalloc((void**)&csr_pivot_pos_dev, nnz * sizeof(unsigned)), "cudaMalloc(csr_pivot_pos_dev)");
    const size_t pair_bytes = A_sym.update_pair_source.size() * sizeof(unsigned);
    if (destination_owned) {
        destination_state = std::make_shared<DestinationGroupState>();
        CUDA_RETURN_ON_ERR(cudaMalloc((void**)&destination_state->lower, A_sym.destination_lower.size() * sizeof(unsigned)), "destination lower allocation");
        CUDA_RETURN_ON_ERR(cudaMalloc((void**)&destination_state->upper, A_sym.destination_upper.size() * sizeof(unsigned)), "destination upper allocation");
        CUDA_RETURN_ON_ERR(cudaMalloc((void**)&destination_state->pointer, A_sym.destination_group_ptr.size() * sizeof(unsigned)), "destination pointer allocation");
        CUDA_RETURN_ON_ERR(cudaMalloc((void**)&destination_state->target, A_sym.destination_target.size() * sizeof(unsigned)), "destination target allocation");
    }
    if (pair_bytes) {
        CUDA_RETURN_ON_ERR(cudaMalloc((void**)&update_pair_source_dev, pair_bytes), "cudaMalloc(update_pair_source)");
        CUDA_RETURN_ON_ERR(cudaMalloc((void**)&update_pair_target_dev, pair_bytes), "cudaMalloc(update_pair_target)");
    }
    const size_t cached_prepare_bytes = A_sym.cached_prepare_sources.size() * sizeof(unsigned);
    const size_t scratch_prepare_bytes = A_sym.scratch_prepare_sources.size() * sizeof(unsigned);
    if (cached_prepare_bytes)
        CUDA_RETURN_ON_ERR(cudaMalloc((void**)&cached_prepare_sources_dev, cached_prepare_bytes), "cached prepare queue allocation");
    if (scratch_prepare_bytes)
        CUDA_RETURN_ON_ERR(cudaMalloc((void**)&scratch_prepare_sources_dev, scratch_prepare_bytes), "scratch prepare queue allocation");
    const size_t task_bytes = A_sym.batch_task_entry.size() * sizeof(unsigned);
    const size_t short_task_bytes = A_sym.short_update_task_index.size() * sizeof(unsigned);
    if (short_task_bytes)
        CUDA_RETURN_ON_ERR(cudaMalloc((void**)&short_update_task_index_dev, short_task_bytes), "cudaMalloc(short_update_task_index)");
    if (task_bytes) {
        CUDA_RETURN_ON_ERR(cudaMalloc((void**)&batch_task_entry_dev, task_bytes), "cudaMalloc(batch_task_entry)");
        CUDA_RETURN_ON_ERR(cudaMalloc((void**)&batch_task_source_dev, task_bytes), "cudaMalloc(batch_task_source)");
        CUDA_RETURN_ON_ERR(cudaMalloc((void**)&batch_task_begin_dev, task_bytes), "cudaMalloc(batch_task_begin)");
        CUDA_RETURN_ON_ERR(cudaMalloc((void**)&batch_task_end_dev, task_bytes), "cudaMalloc(batch_task_end)");
        CUDA_RETURN_ON_ERR(cudaMalloc((void**)&update_pair_begin_dev, task_bytes), "cudaMalloc(update_pair_begin)");
        CUDA_RETURN_ON_ERR(cudaMalloc((void**)&update_pair_end_dev, task_bytes), "cudaMalloc(update_pair_end)");
    }

    if (dense_size) {
        dense_state = std::make_shared<DenseTailState>();
        CUDA_RETURN_ON_ERR(cudaMalloc((void**)&dense_state->positions, A_sym.dense_positions.size() * sizeof(unsigned)), "dense positions allocation");
        CUDA_RETURN_ON_ERR(cudaMalloc((void**)&dense_state->values, size_t(dense_size) * dense_size * sizeof(REAL)), "dense values allocation");
        CUDA_RETURN_ON_ERR(cudaMalloc((void**)&dense_state->info, sizeof(int)), "dense info allocation");
    }
    CUDA_RETURN_ON_ERR(cudaDeviceSynchronize(), "allocation synchronize");
    record("reusable_allocation", phase_start, true);
    phase_start = stamp();
    CUDA_RETURN_ON_ERR(cudaMemcpy(sym_c_ptr_dev, &(A_sym.sym_c_ptr[0]), (n + 1) * sizeof(unsigned), cudaMemcpyHostToDevice),
        "cudaMemcpy(sym_c_ptr_dev)");
    if (dense_size)
        CUDA_RETURN_ON_ERR(cudaMemcpy(dense_state->positions, A_sym.dense_positions.data(), A_sym.dense_positions.size() * sizeof(unsigned), cudaMemcpyHostToDevice), "dense positions upload");
    CUDA_RETURN_ON_ERR(cudaMemcpy(sym_r_idx_dev, &(A_sym.sym_r_idx[0]), nnz * sizeof(unsigned), cudaMemcpyHostToDevice),
        "cudaMemcpy(sym_r_idx_dev)");
    CUDA_RETURN_ON_ERR(cudaMemcpy(csr_pivot_pos_dev, A_sym.csr_pivot_pos.data(), nnz * sizeof(unsigned), cudaMemcpyHostToDevice), "cudaMemcpy(csr_pivot_pos_dev)");
    if (numeric_entries)
        CUDA_RETURN_ON_ERR(cudaMemcpy(numeric_dst_dev, A_sym.numeric_scatter_dst.data(), numeric_entries * sizeof(unsigned), cudaMemcpyHostToDevice), "cudaMemcpy(numeric_destinations)");
    if (task_bytes) {
        CUDA_RETURN_ON_ERR(cudaMemcpy(batch_task_entry_dev, A_sym.batch_task_entry.data(), task_bytes, cudaMemcpyHostToDevice), "cudaMemcpy(batch_task_entry)");
        CUDA_RETURN_ON_ERR(cudaMemcpy(batch_task_source_dev, A_sym.batch_task_source.data(), task_bytes, cudaMemcpyHostToDevice), "cudaMemcpy(batch_task_source)");
        CUDA_RETURN_ON_ERR(cudaMemcpy(batch_task_begin_dev, A_sym.batch_task_begin.data(), task_bytes, cudaMemcpyHostToDevice), "cudaMemcpy(batch_task_begin)");
        CUDA_RETURN_ON_ERR(cudaMemcpy(batch_task_end_dev, A_sym.batch_task_end.data(), task_bytes, cudaMemcpyHostToDevice), "cudaMemcpy(batch_task_end)");
        CUDA_RETURN_ON_ERR(cudaMemcpy(update_pair_begin_dev, A_sym.update_pair_begin.data(), task_bytes, cudaMemcpyHostToDevice), "cudaMemcpy(update_pair_begin)");
        CUDA_RETURN_ON_ERR(cudaMemcpy(update_pair_end_dev, A_sym.update_pair_end.data(), task_bytes, cudaMemcpyHostToDevice), "cudaMemcpy(update_pair_end)");
    }
    if (pair_bytes) {
        CUDA_RETURN_ON_ERR(cudaMemcpy(update_pair_source_dev, A_sym.update_pair_source.data(), pair_bytes, cudaMemcpyHostToDevice), "cudaMemcpy(update_pair_source)");
        CUDA_RETURN_ON_ERR(cudaMemcpy(update_pair_target_dev, A_sym.update_pair_target.data(), pair_bytes, cudaMemcpyHostToDevice), "cudaMemcpy(update_pair_target)");
    }
    if (cached_prepare_bytes)
        CUDA_RETURN_ON_ERR(cudaMemcpy(cached_prepare_sources_dev, A_sym.cached_prepare_sources.data(), cached_prepare_bytes, cudaMemcpyHostToDevice), "cached prepare queue upload");
    if (scratch_prepare_bytes)
        CUDA_RETURN_ON_ERR(cudaMemcpy(scratch_prepare_sources_dev, A_sym.scratch_prepare_sources.data(), scratch_prepare_bytes, cudaMemcpyHostToDevice), "scratch prepare queue upload");
    if (short_task_bytes)
        CUDA_RETURN_ON_ERR(cudaMemcpy(short_update_task_index_dev, A_sym.short_update_task_index.data(), short_task_bytes, cudaMemcpyHostToDevice), "cudaMemcpy(short_update_task_index)");
    if (destination_owned) {
        CUDA_RETURN_ON_ERR(cudaMemcpy(destination_state->lower, A_sym.destination_lower.data(), A_sym.destination_lower.size() * sizeof(unsigned), cudaMemcpyHostToDevice), "destination lower upload");
        CUDA_RETURN_ON_ERR(cudaMemcpy(destination_state->upper, A_sym.destination_upper.data(), A_sym.destination_upper.size() * sizeof(unsigned), cudaMemcpyHostToDevice), "destination upper upload");
        CUDA_RETURN_ON_ERR(cudaMemcpy(destination_state->pointer, A_sym.destination_group_ptr.data(), A_sym.destination_group_ptr.size() * sizeof(unsigned), cudaMemcpyHostToDevice), "destination pointer upload");
        CUDA_RETURN_ON_ERR(cudaMemcpy(destination_state->target, A_sym.destination_target.data(), A_sym.destination_target.size() * sizeof(unsigned), cudaMemcpyHostToDevice), "destination target upload");
    }
    record("structural_upload", phase_start, true);
    phase_start = stamp();
    if (device_assembly) {
        if (numeric_entries)
            CUDA_RETURN_ON_ERR(cudaMemcpy(numeric_values_dev, A_sym.numeric_packed.data(), numeric_entries * sizeof(REAL), cudaMemcpyHostToDevice), "cudaMemcpy(numeric_packed)");
    } else {
        CUDA_RETURN_ON_ERR(cudaMemcpy(val_dev, &(A_sym.val[0]), nnz * sizeof(REAL), cudaMemcpyHostToDevice), "cudaMemcpy(val_dev)");
    }
    record("numeric_upload", phase_start, false);
    if (device_assembly) {
        phase_start = stamp();
        CUDA_RETURN_ON_ERR(cudaMemset(val_dev, 0, nnz * sizeof(REAL)), "reset numeric values");
        if (numeric_entries) initialize_sparse_values<<<(numeric_entries + 255) / 256, 256>>>(
            val_dev, numeric_dst_dev, numeric_values_dev, static_cast<unsigned>(numeric_entries));
        CUDA_RETURN_ON_ERR(cudaGetLastError(), "numeric scatter launch");
        CUDA_RETURN_ON_ERR(cudaDeviceSynchronize(), "numeric scatter completion");
        record("numeric_preparation", phase_start, false);
    }
    phase_start = stamp();
    CUDA_RETURN_ON_ERR(cudaMemcpy(l_col_ptr_dev, &(A_sym.l_col_ptr[0]), n * sizeof(unsigned), cudaMemcpyHostToDevice),
        "cudaMemcpy(l_col_ptr_dev)");
    CUDA_RETURN_ON_ERR(cudaMemcpy(csr_r_ptr_dev, &(A_sym.csr_r_ptr[0]), (n + 1) * sizeof(unsigned), cudaMemcpyHostToDevice),
        "cudaMemcpy(csr_r_ptr_dev)");
    CUDA_RETURN_ON_ERR(cudaMemcpy(csr_c_idx_dev, &(A_sym.csr_c_idx[0]), nnz * sizeof(unsigned), cudaMemcpyHostToDevice),
        "cudaMemcpy(csr_c_idx_dev)");
    CUDA_RETURN_ON_ERR(cudaMemcpy(csr_diag_ptr_dev, &(A_sym.csr_diag_ptr[0]), n * sizeof(unsigned), cudaMemcpyHostToDevice),
        "cudaMemcpy(csr_diag_ptr_dev)");
    CUDA_RETURN_ON_ERR(cudaMemcpy(level_idx_dev, &(A_sym.level_idx[0]), n * sizeof(int), cudaMemcpyHostToDevice),
        "cudaMemcpy(level_idx_dev)");

    record("structural_upload", phase_start, true);
    phase_start = stamp();
    for (int j = 0; j < Nstreams; ++j) {
        CUDA_RETURN_ON_ERR(cudaStreamCreate(&streams[j]), "cudaStreamCreate");
        stream_created[j] = true;
    }

    size_t free_bytes = 0, total_bytes = 0;
    CUDA_RETURN_ON_ERR(cudaMemGetInfo(&free_bytes, &total_bytes), "cudaMemGetInfo");
    (void)total_bytes;

    const size_t reserve_bytes = 4ull * 1024ull * 1024ull * 1024ull;
    size_t max_tmp_mem_size = (free_bytes > reserve_bytes) ? (free_bytes - reserve_bytes) : (free_bytes / 2);
    const unsigned first_level_cols = (A_sym.level_ptr.size() > 1) ?
        static_cast<unsigned>(A_sym.level_ptr[1] - A_sym.level_ptr[0]) : 1u;
    const size_t good_tmp_mem_choice = sizeof(REAL) * size_t(n) * size_t(first_level_cols);
    const size_t bytes_per_column = sizeof(REAL) * size_t(n);
    if (max_tmp_mem_size < bytes_per_column)
        max_tmp_mem_size = bytes_per_column;

    // Each value slot also owns a four-byte tag; reserve memory for both.
    max_tmp_mem_size = max_tmp_mem_size / (sizeof(REAL) + sizeof(unsigned)) * sizeof(REAL);
    if (max_tmp_mem_size < bytes_per_column) max_tmp_mem_size = bytes_per_column;
    unsigned TMPMEMNUM = 0;
    if (good_tmp_mem_choice <= max_tmp_mem_size)
        TMPMEMNUM = first_level_cols;
    else
        TMPMEMNUM = static_cast<unsigned>(max_tmp_mem_size / bytes_per_column);
    if (TMPMEMNUM == 0)
        TMPMEMNUM = 1;

    // Preserve source concurrency as n grows, but avoid clearing 8 GiB
    // for low-dimensional problems that need far fewer bytes per source.
    const size_t min_scratch_bytes = 1ull * 1024ull * 1024ull * 1024ull;
    const size_t max_scratch_bytes = 8ull * 1024ull * 1024ull * 1024ull;
    size_t scratch_budget = bytes_per_column * size_t(1024);
    if (scratch_budget < min_scratch_bytes) scratch_budget = min_scratch_bytes;
    if (scratch_budget > max_scratch_bytes) scratch_budget = max_scratch_bytes;
    unsigned scratch_slots = static_cast<unsigned>(scratch_budget / bytes_per_column);
    if (scratch_slots < static_cast<unsigned>(Nstreams)) scratch_slots = Nstreams;
    if (TMPMEMNUM > scratch_slots) TMPMEMNUM = scratch_slots;
    const size_t tmp_bytes = size_t(TMPMEMNUM) * size_t(n) * sizeof(REAL);
    const size_t tag_bytes = size_t(TMPMEMNUM) * size_t(n) * sizeof(unsigned);
    CUDA_RETURN_ON_ERR(cudaMalloc((void**)&tmpMem, tmp_bytes), "cudaMalloc(tmpMem)");
    CUDA_RETURN_ON_ERR(cudaMalloc((void**)&scratch_tag_dev, tag_bytes), "cudaMalloc(scratch_tag)");
    CUDA_RETURN_ON_ERR(cudaDeviceSynchronize(), "scratch allocation synchronize");
    record("reusable_allocation", phase_start, true);
    phase_start = stamp();
    CUDA_RETURN_ON_ERR(cudaMemset(scratch_tag_dev, 0, tag_bytes), "cudaMemset(scratch_tag)");

    CUDA_RETURN_ON_ERR(cudaDeviceSynchronize(), "scratch reset synchronize");
    record("scratch_reset", phase_start, false);

    // calculate 1-norm of A and perturbation value for perturbation
    REAL pert = 0;
    if (PERTURB)
    {
        REAL norm_A = 0;
        for (unsigned i = 0; i < n; ++i)
        {
            REAL tmp = 0;
            for (unsigned j = A_sym.sym_c_ptr[i]; j < A_sym.sym_c_ptr[i+1]; ++j)
                tmp += fabs(A_sym.val[j]);
            if (norm_A < tmp)
                norm_A = tmp;
        }
        pert = (REAL)3.45e-4 * norm_A;
        out << "Gaussian elimination with static pivoting (GESP)..." << endl;
        out << "1-Norm of A matrix is " << norm_A << ", Perturbation value is " << pert << endl;
    }

    bool first_kernel = true;
    bool first_kernel_ok = true;
    double first_start = 0;
    const auto before_launch = [&]() { if (first_kernel && report) first_start = stamp(); };
    const auto after_launch = [&]() {
        if (first_kernel && report) {
            first_kernel_ok = cudaCheck(cudaGetLastError(), "first kernel launch") &&
                              cudaCheck(cudaDeviceSynchronize(), "first kernel synchronize");
            if (first_kernel_ok) record("first_kernel_startup", first_start, false);
            first_kernel = false;
        }
    };
    auto launch_batched_level = [&](unsigned level_head, int level_size, unsigned warps_per_block,
                                    cudaStream_t launch_stream, bool capturing) {
        dim3 dimBlock(warps_per_block * 32, 1);
        size_t mem_size = 0; // coefficients are broadcast within each warp

        int remaining = level_size;
        unsigned chunk_idx = 0;
        while (remaining > 0) {
            unsigned rest_col = static_cast<unsigned>(remaining) > TMPMEMNUM ?
                TMPMEMNUM : static_cast<unsigned>(remaining);
            dim3 dimGrid(rest_col, 1);
            int in_level_pos = static_cast<int>(chunk_idx * TMPMEMNUM);
            if (A_sym.batch_task_ptr[level_head + level_size] > A_sym.batch_task_ptr[level_head]) {
                const unsigned source_head = level_head + in_level_pos;
                const unsigned task_head = A_sym.batch_task_ptr[source_head];
                const unsigned tasks = A_sym.batch_task_ptr[source_head + rest_col] - task_head;
                const unsigned short_head = A_sym.short_update_task_ptr[source_head];
                const unsigned short_tasks = A_sym.short_update_task_ptr[source_head + rest_col] - short_head;
                const bool packed_short = short_tasks >= 64u * unsigned(std::max(1, deviceProp.multiProcessorCount));
                if (PERTURB) {
                    // Preserve the original perturbation of every source,
                    // including empty-L columns omitted from normal queues.
                    if (!capturing) before_launch();
                    RL_batch_prepare<<<rest_col, 256, 0, launch_stream>>>(sym_c_ptr_dev, sym_r_idx_dev, val_dev,
                        l_col_ptr_dev, level_idx_dev, scratch_tag_dev, tmpMem, n, source_head, PERTURB, pert, nullptr, 0);
                    if (!capturing) after_launch();
                } else {
                    const unsigned cache_first = A_sym.cached_prepare_ptr[source_head];
                    const unsigned cache_end = A_sym.cached_prepare_ptr[source_head + rest_col];
                    const unsigned scatter_first = A_sym.scratch_prepare_ptr[source_head];
                    const unsigned scatter_end = A_sym.scratch_prepare_ptr[source_head + rest_col];
                    if (cache_end > cache_first) {
                        if (!capturing) before_launch();
                        RL_cached_source_prepare<<<(cache_end - cache_first + 3) / 4, 4 * deviceProp.warpSize, 0, launch_stream>>>(
                            sym_c_ptr_dev, val_dev, l_col_ptr_dev, level_idx_dev, cached_prepare_sources_dev, cache_first, cache_end);
                        if (!capturing) after_launch();
                    }
                    if (scatter_end > scatter_first) {
                        if (!capturing) before_launch();
                        RL_batch_prepare<<<scatter_end - scatter_first, 256, 0, launch_stream>>>(sym_c_ptr_dev, sym_r_idx_dev, val_dev,
                            l_col_ptr_dev, level_idx_dev, scratch_tag_dev, tmpMem, n, source_head, false, pert,
                            scratch_prepare_sources_dev, scatter_first);
                        if (!capturing) after_launch();
                    }
                }
                if (tasks) {
                    if (!capturing) before_launch();
                    RL_update_tasks<<<(tasks + 3) / 4, 4 * deviceProp.warpSize, 0, launch_stream>>>(sym_c_ptr_dev,
                    sym_r_idx_dev, val_dev, l_col_ptr_dev, csr_c_idx_dev, level_idx_dev,
                    csr_pivot_pos_dev, batch_task_entry_dev, batch_task_source_dev, batch_task_begin_dev, batch_task_end_dev, update_pair_begin_dev, update_pair_end_dev, update_pair_source_dev, update_pair_target_dev, scratch_tag_dev, tmpMem, n, source_head, task_head, task_head + tasks, packed_short);
                    if (!capturing) after_launch();
                }
                if (packed_short) RL_update_short_tasks<<<(short_tasks + 15) / 16, 256, 0, launch_stream>>>(
                    val_dev, csr_pivot_pos_dev, batch_task_entry_dev, short_update_task_index_dev,
                    update_pair_begin_dev, update_pair_end_dev, update_pair_source_dev, update_pair_target_dev,
                    short_head, short_head + short_tasks);
                remaining -= static_cast<int>(rest_col);
                ++chunk_idx;
                continue;
            }
            if (!capturing) before_launch();
            if (!PERTURB)
                RL<<<dimGrid, dimBlock, mem_size, launch_stream>>>(sym_c_ptr_dev,
                                                    sym_r_idx_dev,
                                                    val_dev,
                                                    l_col_ptr_dev,
                                                    csr_r_ptr_dev,
                                                    csr_c_idx_dev,
                                                    csr_diag_ptr_dev,
                                                    csr_pivot_pos_dev,
                                                    level_idx_dev,
                                                    scratch_tag_dev,
                                                    tmpMem,
                                                    n,
                                                    level_head,
                                                    in_level_pos);
            else
                RL_perturb<<<dimGrid, dimBlock, mem_size, launch_stream>>>(sym_c_ptr_dev,
                                                            sym_r_idx_dev,
                                                            val_dev,
                                                            l_col_ptr_dev,
                                                            csr_r_ptr_dev,
                                                            csr_c_idx_dev,
                                                            csr_diag_ptr_dev,
                                                            csr_pivot_pos_dev,
                                                            level_idx_dev,
                                                            scratch_tag_dev,
                                                            tmpMem,
                                                            n,
                                                            level_head,
                                                            in_level_pos,
                                                            pert);
            if (!capturing) after_launch();
            remaining -= static_cast<int>(rest_col);
            ++chunk_idx;
        }
    };

    const auto launch_destination_level = [&](unsigned lv, cudaStream_t stream, bool capturing) {
        const unsigned source_first = A_sym.level_ptr[lv], source_end = A_sym.level_ptr[lv + 1];
        const unsigned count = source_end - source_first;
        if (count) {
            if (!capturing) before_launch();
            destination_source_normalize<<<(count + 3) / 4, 4 * deviceProp.warpSize, 0, stream>>>(
                val_dev, sym_c_ptr_dev, l_col_ptr_dev, level_idx_dev, source_first, source_end);
            if (!capturing) after_launch();
        }
        const unsigned first = A_sym.destination_level_ptr[lv], end = A_sym.destination_level_ptr[lv + 1];
        if (end > first) destination_group_reduce<<<(end - first + 255) / 256, 256, 0, stream>>>(
            val_dev, destination_state->pointer, destination_state->target,
            destination_state->lower, destination_state->upper, first, end);
    };

    // Build only a reusable schedule: stream capture enqueues no numerical
    // execution and reads no matrix values. Numeric work starts below.
    const double graph_prepare_start = stamp();
    std::vector<unsigned> persistent_end(num_lev, 0);
    std::vector<unsigned char> narrow_ok(num_lev, 0), persistent_covered(num_lev, 0);
    for (unsigned lv = 1; lv < num_lev; ++lv) {
        if (dense_level(lv)) continue;
        const unsigned head = A_sym.level_ptr[lv];
        const unsigned count = A_sym.level_ptr[lv + 1] - head;
        bool ok = count > 0 && count <= 2;
        for (unsigned pos = head; ok && pos < head + count; ++pos) {
            const unsigned col = A_sym.level_idx[pos];
            ok = A_sym.sym_c_ptr[col + 1] - A_sym.l_col_ptr[col] - 1 <= 256 &&
                 A_sym.csr_r_ptr[col + 1] - A_sym.csr_diag_ptr[col] - 1 <= 32;
            for (unsigned entry = A_sym.csr_diag_ptr[col] + 1; ok && entry < A_sym.csr_r_ptr[col + 1]; ++entry) {
                const unsigned target = A_sym.csr_c_idx[entry];
                ok = A_sym.sym_c_ptr[target + 1] - A_sym.sym_c_ptr[target] <= 512;
            }
        }
        narrow_ok[lv] = ok;
    }
    unsigned persistent_chains = 0, persistent_levels = 0;
    if (!PERTURB) for (unsigned head = 1; head < num_lev;) {
        if (!narrow_ok[head]) { ++head; continue; }
        unsigned end = head + 1;
        while (end < num_lev && narrow_ok[end] && end - head < 256) ++end;
        if (end - head >= 8) {
            persistent_end[head] = end;
            for (unsigned lv = head; lv < end; ++lv) persistent_covered[lv] = 1;
            ++persistent_chains;
            persistent_levels += end - head;
        }
        head = end;
    }
    if (persistent_chains) {
        CUDA_RETURN_ON_ERR(cudaMalloc((void**)&level_ptr_dev, (num_lev + 1) * sizeof(int)), "cudaMalloc(level_ptr)");
        CUDA_RETURN_ON_ERR(cudaMemcpy(level_ptr_dev, A_sym.level_ptr.data(), (num_lev + 1) * sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpy(level_ptr)");
    }
    out << "Persistent narrow chains: " << persistent_chains << ", levels: " << persistent_levels << endl;
    std::vector<unsigned char> graph_eligible(num_lev, 0);
    for (unsigned lv = 1; lv < num_lev; ++lv) {
        const unsigned head = A_sym.level_ptr[lv];
        const unsigned count = A_sym.level_ptr[lv + 1] - head;
        bool small = count > 0 && count <= Nstreams;
        if (small) for (unsigned pos = head; pos < head + count; ++pos) {
            const unsigned col = A_sym.level_idx[pos];
            if (A_sym.sym_c_ptr[col + 1] - A_sym.l_col_ptr[col] - 1 > 2048u ||
                A_sym.csr_r_ptr[col + 1] - A_sym.csr_diag_ptr[col] - 1 > 64u) {
                small = false; break;
            }
        }
        graph_eligible[lv] = !dense_level(lv) && !persistent_covered[lv] && (count > Nstreams || small ||
            A_sym.batch_task_ptr[head + count] > A_sym.batch_task_ptr[head]);
    }
    if (destination_owned) {
        CUDA_RETURN_ON_ERR(cudaStreamCreateWithFlags(&capture_stream, cudaStreamNonBlocking), "destination capture stream");
        for (unsigned head = 1; head < num_lev;) {
            const unsigned end = std::min(head + 128u, num_lev);
            CUDA_RETURN_ON_ERR(cudaStreamBeginCapture(capture_stream, cudaStreamCaptureModeGlobal), "destination capture begin");
            capture_active = true;
            for (unsigned lv = head; lv < end; ++lv) launch_destination_level(lv, capture_stream, true);
            const cudaError_t status = cudaStreamEndCapture(capture_stream, &level_graph[head]);
            capture_active = false;
            CUDA_RETURN_ON_ERR(status, "destination capture end");
            CUDA_RETURN_ON_ERR(cudaGraphInstantiate(&level_exec[head], level_graph[head], nullptr, nullptr, 0), "destination graph instantiate");
            graph_end[head] = end; ++graph_segments; graph_levels += end - head; head = end;
        }
    }
    if (!PERTURB && !destination_owned) {
        CUDA_RETURN_ON_ERR(cudaStreamCreateWithFlags(&capture_stream, cudaStreamNonBlocking), "graph capture stream");
        if (dense_size) {
            CUDA_RETURN_ON_ERR(cudaStreamBeginCapture(capture_stream, cudaStreamCaptureModeGlobal), "dense panel capture begin");
            capture_active = true;
            unsigned kernels = 0;
            for (unsigned first = 0; first < dense_size; first += 16) {
                const unsigned end = std::min(first + 16u, dense_size);
                dense_panel_factor<<<1, 256, 0, capture_stream>>>(dense_state->values, dense_size, first, end, dense_state->info);
                ++kernels;
                if (end < dense_size) {
                    const unsigned tail = dense_size - end;
                    dense_panel_upper<<<tail, deviceProp.warpSize, 0, capture_stream>>>(
                        dense_state->values, dense_size, first, end, dense_state->info);
                    dense_panel_update<<<dim3((tail + 15) / 16, (tail + 15) / 16), dim3(16, 16), 0, capture_stream>>>(
                        dense_state->values, dense_size, first, end, dense_state->info);
                    kernels += 2;
                }
            }
            const cudaError_t status = cudaStreamEndCapture(capture_stream, &dense_state->graph);
            capture_active = false;
            CUDA_RETURN_ON_ERR(status, "dense panel capture end");
            CUDA_RETURN_ON_ERR(cudaGraphInstantiate(&dense_state->exec, dense_state->graph, nullptr, nullptr, 0), "dense panel instantiate");
            out << "Dense panel graph: " << dense_size << ", panel: 16, numerical kernels: " << kernels << endl;
        }
        for (unsigned head = 1; head < num_lev;) {
            if (!graph_eligible[head]) { ++head; continue; }
            unsigned end = head + 1;
            // Bound graph construction complexity while keeping level order.
            while (end < num_lev && graph_eligible[end] && end - head < 128u) ++end;
            if (end - head >= 8) {
                CUDA_RETURN_ON_ERR(cudaStreamBeginCapture(capture_stream, cudaStreamCaptureModeGlobal), "graph capture begin");
                capture_active = true;
                for (unsigned lv = head; lv < end; ++lv) {
                    const unsigned count = A_sym.level_ptr[lv + 1] - A_sym.level_ptr[lv];
                    const unsigned warps = count > 896 ? 2 : (count > 448 ? 4 : 32);
                    launch_batched_level(A_sym.level_ptr[lv], count, warps, capture_stream, true);
                }
                const cudaError_t capture_status = cudaStreamEndCapture(capture_stream, &level_graph[head]);
                capture_active = false;
                CUDA_RETURN_ON_ERR(capture_status, "graph capture end");
                CUDA_RETURN_ON_ERR(cudaGraphInstantiate(&level_exec[head], level_graph[head], nullptr, nullptr, 0), "graph instantiate");
                graph_end[head] = end;
                ++graph_segments;
                graph_levels += end - head;
            }
            head = end;
        }
    }
    CUDA_RETURN_ON_ERR(cudaDeviceSynchronize(), "graph preparation synchronize");
    record("structure_preparation", graph_prepare_start, true);
    const double factor_start = stamp();

    unsigned expanded_batch_levels = 0;
    bool pending_batched = false;
    if (destination_owned) {
        launch_destination_level(0, 0, false);
        if (!first_kernel_ok) { cleanup(); return; }
        CUDA_RETURN_ON_ERR(cudaGetLastError(), "destination level zero launch");
        for (unsigned head = 1; head < num_lev; head = graph_end[head])
            CUDA_RETURN_ON_ERR(cudaGraphLaunch(level_exec[head], 0), "destination graph launch");
        pending_batched = true;
    }
    for (unsigned i = 0; !destination_owned && i < num_lev; ++i)
    {
        int lev_size = A_sym.level_ptr[i + 1] - A_sym.level_ptr[i];
        if (lev_size <= 0)
            continue;
        if (dense_size && int(i) == A_sym.dense_first_level) {
            CUDA_RETURN_ON_ERR(cudaDeviceSynchronize(), "dense predecessor completion");
            const unsigned entries = dense_size * dense_size;
            CUDA_RETURN_ON_ERR(cudaMemset(dense_state->info, 0, sizeof(int)), "dense info reset");
            dense_tail_pack<<<(entries + 255) / 256, 256>>>(val_dev, dense_state->positions, dense_state->values, entries);
            CUDA_RETURN_ON_ERR(cudaGraphLaunch(dense_state->exec, 0), "dense panel graph launch");
            dense_tail_unpack<<<(entries + 255) / 256, 256>>>(val_dev, dense_state->positions, dense_state->values, entries);
            CUDA_RETURN_ON_ERR(cudaGetLastError(), "dense pack/unpack launches");
            int info = -1;
            CUDA_RETURN_ON_ERR(cudaMemcpy(&info, dense_state->info, sizeof(int), cudaMemcpyDeviceToHost), "dense info download");
            if (info) { err << "dense zero pivot: " << info << endl; cleanup(); return; }
            pending_batched = true;
            i = A_sym.dense_end_level - 1;
            continue;
        }
        if (persistent_end[i]) {
            RL_narrow_chain<<<1, 1024>>>(sym_c_ptr_dev, sym_r_idx_dev, val_dev, l_col_ptr_dev,
                csr_r_ptr_dev, csr_c_idx_dev, csr_diag_ptr_dev, csr_pivot_pos_dev, level_idx_dev, scratch_tag_dev,
                tmpMem, n, level_ptr_dev, i, persistent_end[i]);
            CUDA_RETURN_ON_ERR(cudaGetLastError(), "persistent chain launch");
            pending_batched = true;
            i = persistent_end[i] - 1;
            continue;
        }
        if (level_exec[i]) {
            CUDA_RETURN_ON_ERR(cudaGraphLaunch(level_exec[i], 0), "graph launch");
            pending_batched = true;
            i = graph_end[i] - 1;
            continue;
        }

        // A batched warp handles each sub-column. Keep the existing wide-
        // column backend when this would leave too little parallelism.
        bool batch_small = lev_size <= Nstreams;
        bool fits_previous_small_rule = true;
        if (batch_small) {
            for (int j = 0; j < lev_size; ++j) {
                const unsigned col = A_sym.level_idx[A_sym.level_ptr[i] + j];
                const unsigned lower = A_sym.sym_c_ptr[col + 1] - A_sym.l_col_ptr[col] - 1;
                const unsigned upper = A_sym.csr_r_ptr[col + 1] - A_sym.csr_diag_ptr[col] - 1;
                if (lower > 1024u || upper > 32u) fits_previous_small_rule = false;
                if (lower > 2048u || upper > 64u) { batch_small = false; break; }
            }
        }

        // Default-stream batched levels are ordered without a host round trip.
        // Cross into the multistream backend only after prior levels finish.
        const bool task_level = A_sym.batch_task_ptr[A_sym.level_ptr[i + 1]] >
                                A_sym.batch_task_ptr[A_sym.level_ptr[i]];
        const bool batched_level = lev_size > Nstreams || batch_small || task_level;
        if (batch_small && !fits_previous_small_rule) ++expanded_batch_levels;
        if (!batched_level && pending_batched) {
            CUDA_RETURN_ON_ERR(cudaDeviceSynchronize(), "batch-to-stream dependency");
            pending_batched = false;
        }

        if (lev_size > 896) {
            launch_batched_level(A_sym.level_ptr[i], lev_size, 2, 0, false);
        }
        else if (lev_size > 448) {
            launch_batched_level(A_sym.level_ptr[i], lev_size, 4, 0, false);
        }
        else if (lev_size > Nstreams || batch_small || task_level) {
            launch_batched_level(A_sym.level_ptr[i], lev_size, 32, 0, false);
        }
        else {
            // Small levels are mapped to one stream per column to reduce launch overhead.
            for (int offset = 0; offset < lev_size; offset += Nstreams) {
                for (int j = 0; j < Nstreams; j++) {
                    if (j + offset < lev_size) {
                        const unsigned currentCol = A_sym.level_idx[A_sym.level_ptr[i] + j + offset];
                        const unsigned subMatSize = A_sym.csr_r_ptr[currentCol + 1]
                            - A_sym.csr_diag_ptr[currentCol] - 1;

                        before_launch();
                        if (!PERTURB)
                            RL_onecol_factorizeCurrentCol<<<1, 1024, 0, streams[j]>>>(sym_c_ptr_dev,
                                                                                        sym_r_idx_dev,
                                                                                        val_dev,
                                                                                        l_col_ptr_dev,
                                                                                        currentCol,
                                                                                        scratch_tag_dev,
                                                                                        tmpMem,
                                                                                        j,
                                                                                        n);
                        else
                            RL_onecol_factorizeCurrentCol_perturb<<<1, 1024, 0, streams[j]>>>(sym_c_ptr_dev,
                                                                                                sym_r_idx_dev,
                                                                                                val_dev,
                                                                                                l_col_ptr_dev,
                                                                                                currentCol,
                                                                                                scratch_tag_dev,
                                                                                                tmpMem,
                                                                                                j,
                                                                                                n,
                                                                                                pert);
                        after_launch();
                        if (subMatSize > 0)
                            RL_onecol_updateSubmat<<<subMatSize, 1024, 0, streams[j]>>>(sym_c_ptr_dev,
                                                                                          sym_r_idx_dev,
                                                                                          val_dev,
                                                                                          csr_c_idx_dev,
                                                                                          csr_diag_ptr_dev,
                                                                                          csr_pivot_pos_dev,
                                                                                          currentCol,
                                                                                          scratch_tag_dev,
                                                                                          tmpMem,
                                                                                          j,
                                                                                          n);
                    }
                }
            }
        }
        if (!first_kernel_ok) { cleanup(); return; }
        CUDA_RETURN_ON_ERR(cudaGetLastError(), "kernel launch");
        if (batched_level) {
            pending_batched = true;
        } else {
            CUDA_RETURN_ON_ERR(cudaDeviceSynchronize(), "multistream level completion");
        }
    }

    // Include every queued operation in fully charged factorization timing.
    if (pending_batched)
        CUDA_RETURN_ON_ERR(cudaDeviceSynchronize(), "batched factorization completion");
    record("numeric_factorization", factor_start, false);
    phase_start = stamp();
    CUDA_RETURN_ON_ERR(cudaMemcpy(&(A_sym.val[0]), val_dev, nnz * sizeof(REAL), cudaMemcpyDeviceToHost),
        "cudaMemcpy(A_sym.val)");
    CUDA_RETURN_ON_ERR(cudaEventRecord(stop, 0), "cudaEventRecord(stop)");
    CUDA_RETURN_ON_ERR(cudaEventSynchronize(stop), "cudaEventSynchronize(stop)");
    CUDA_RETURN_ON_ERR(cudaEventElapsedTime(&time, start, stop), "cudaEventElapsedTime");

    record("numeric_download", phase_start, false);
    out << "Total GPU time: " << time << " ms" << endl;
    out << "Additional small levels batched: " << expanded_batch_levels << endl;
    out << "Scratch slots: " << TMPMEMNUM << ", bytes: " << tmp_bytes << endl;
    out << "Graph segments: " << graph_segments << ", captured levels: " << graph_levels << endl;
    out << "Device assembly: " << device_assembly << ", packed entries: " << numeric_entries << "/" << nnz << endl;
    out << "Registered factor receive: " << factor_host_registered << endl;

#ifdef GLU_DEBUG
    // check NaN elements
    unsigned err_find = 0;
    for(unsigned i = 0; i < nnz; i++)
        if(isnan(A_sym.val[i]) || isinf(A_sym.val[i]))
            err_find++;

    if (err_find != 0)
        err << "LU data check: NaN/Inf found." << endl;
#endif
    // Hold completed resources until CPU solve finishes; cleanup remains fully charged.
    void *factor_receive_pointer = A_sym.val.data();
    A_sym.gpu_resources = std::shared_ptr<void>(new int(0), [=](void *token) mutable {
        dense_state.reset();
        destination_state.reset();
        if (capture_active) {
            cudaGraph_t abandoned = nullptr;
            cudaStreamEndCapture(capture_stream, &abandoned);
            if (abandoned) cudaGraphDestroy(abandoned);
            capture_active = false;
        }
        for (unsigned i = 0; i < num_lev; ++i) {
            if (level_exec[i]) cudaGraphExecDestroy(level_exec[i]);
            if (level_graph[i]) cudaGraphDestroy(level_graph[i]);
        }
        if (capture_stream) cudaStreamDestroy(capture_stream);
        for (int j = 0; j < Nstreams; ++j) {
            if (stream_created[j])
                cudaStreamDestroy(streams[j]);
        }
        if (tmpMem != nullptr)
            cudaFree(tmpMem);
        if (scratch_tag_dev != nullptr) cudaFree(scratch_tag_dev);
        if (sym_c_ptr_dev != nullptr)
            cudaFree(sym_c_ptr_dev);
        if (sym_r_idx_dev != nullptr)
            cudaFree(sym_r_idx_dev);
        if (val_dev != nullptr)
            cudaFree(val_dev);
        if (numeric_values_dev) cudaFree(numeric_values_dev);
        if (numeric_dst_dev) cudaFree(numeric_dst_dev);
        if (l_col_ptr_dev != nullptr)
            cudaFree(l_col_ptr_dev);
        if (csr_c_idx_dev != nullptr)
            cudaFree(csr_c_idx_dev);
        if (csr_r_ptr_dev != nullptr)
            cudaFree(csr_r_ptr_dev);
        if (csr_diag_ptr_dev != nullptr)
            cudaFree(csr_diag_ptr_dev);
        if (batch_task_entry_dev != nullptr) cudaFree(batch_task_entry_dev);
        if (batch_task_source_dev != nullptr) cudaFree(batch_task_source_dev);
        if (batch_task_begin_dev != nullptr) cudaFree(batch_task_begin_dev);
        if (batch_task_end_dev != nullptr) cudaFree(batch_task_end_dev);
        if (update_pair_begin_dev != nullptr) cudaFree(update_pair_begin_dev);
        if (update_pair_end_dev != nullptr) cudaFree(update_pair_end_dev);
        if (update_pair_source_dev != nullptr) cudaFree(update_pair_source_dev);
        if (update_pair_target_dev != nullptr) cudaFree(update_pair_target_dev);
        if (short_update_task_index_dev != nullptr) cudaFree(short_update_task_index_dev);
        if (cached_prepare_sources_dev) cudaFree(cached_prepare_sources_dev);
        if (scratch_prepare_sources_dev) cudaFree(scratch_prepare_sources_dev);
        if (csr_pivot_pos_dev != nullptr) cudaFree(csr_pivot_pos_dev);
        if (level_idx_dev != nullptr)
            cudaFree(level_idx_dev);
        if (level_ptr_dev != nullptr) cudaFree(level_ptr_dev);
        if (start != nullptr)
            cudaEventDestroy(start);
        if (stop != nullptr)
            cudaEventDestroy(stop);
        if (factor_host_registered) {
            cudaHostUnregister(factor_receive_pointer);
            factor_host_registered = false;
        }
        delete static_cast<int*>(token);
    });
#undef CUDA_RETURN_ON_ERR
}
