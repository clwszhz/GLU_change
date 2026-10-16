#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include "numeric.h"
#include <cmath>

using namespace std;

struct DestinationGroupState {
    unsigned *lower = nullptr, *upper = nullptr, *pointer = nullptr, *target = nullptr;
    unsigned short *upper_delta = nullptr;
    unsigned *normalize_lower = nullptr, *normalize_diagonal = nullptr;
    unsigned *normalize_level = nullptr, *group_level = nullptr, *heavy_level = nullptr;
    ~DestinationGroupState() {
        if (lower) cudaFree(lower); if (upper) cudaFree(upper);
        if (upper_delta) cudaFree(upper_delta);
        if (pointer) cudaFree(pointer); if (target) cudaFree(target);
        if (normalize_lower) cudaFree(normalize_lower); if (normalize_diagonal) cudaFree(normalize_diagonal);
        if (normalize_level) cudaFree(normalize_level);
        if (group_level) cudaFree(group_level);
        if (heavy_level) cudaFree(heavy_level);
    }
};

__global__ void destination_entry_normalize(REAL *values, const unsigned *lower,
        const unsigned *diagonal, unsigned count) {
    const unsigned p=blockIdx.x*blockDim.x+threadIdx.x;
    if (p<count) values[lower[p]] /= values[diagonal[p]];
}

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

__global__ void destination_group_reduce_relative(REAL *values, const unsigned *pointer,
        const unsigned *target, const unsigned *lower, const unsigned short *upper_delta,
        unsigned first, unsigned end) {
    static_assert(sizeof(unsigned short)==2,"Relative product indices require16-bit unsigned short");
    const unsigned group=first+blockIdx.x*blockDim.x+threadIdx.x;
    if(group>=end) return;
    const unsigned position=target[group];
    REAL value=values[position];
    for(unsigned p=pointer[group];p<pointer[group+1];++p)
        value-=values[lower[p]]*values[position-unsigned(upper_delta[p])];
    values[position]=value;
}

// Every physical warp participates, including lanes with no tail product.
// Reassociation stays in FP64 and is checked by the unchanged evaluator.
template<bool Packed>
__global__ void destination_group_reduce_warp(REAL *values, const unsigned *pointer,
        const unsigned *target, const unsigned *lower, const unsigned *upper,
        const unsigned short *upper_delta, unsigned first, unsigned end) {
    const unsigned group=first+(blockIdx.x*blockDim.x+threadIdx.x)/warpSize;
    if (group>=end) return;
    const unsigned lane=threadIdx.x%warpSize, position=target[group];
    REAL sum=REAL(0);
    for (unsigned p=pointer[group]+lane;p<pointer[group+1];p+=warpSize) {
        const unsigned u=Packed?position-unsigned(upper_delta[p]):upper[p];
        sum+=values[lower[p]]*values[u];
    }
    for (unsigned offset=warpSize/2;offset;offset/=2)
        sum+=__shfl_down_sync(~0ULL,sum,offset,warpSize);
    if (lane==0) values[position]-=sum;
}

// Each original phase here fits one 256-thread block. All threads stay alive
// through every barrier; no synchronization with another block is required.
template<bool Packed>
__global__ void destination_gather_level_run(REAL *values,
        const unsigned *normal_level,const unsigned *group_level,const unsigned *heavy_level,
        const unsigned *normal_lower,const unsigned *normal_diagonal,
        const unsigned *pointer,const unsigned *target,const unsigned *lower,
        const unsigned *upper,const unsigned short *upper_delta,unsigned first,unsigned end) {
    const unsigned tid=threadIdx.x,lane=tid%warpSize,warp=tid/warpSize;
    for(unsigned lv=first;lv<end;++lv) {
        const unsigned na=normal_level[lv],nb=normal_level[lv+1];
        if(tid<nb-na) {
            const unsigned p=na+tid;
            values[normal_lower[p]]/=values[normal_diagonal[p]];
        }
        __syncthreads();
        const unsigned ga=group_level[lv],gb=group_level[lv+1];
        const unsigned split=heavy_level?heavy_level[lv]:gb;
        const unsigned short_group=ga+tid;
        if(short_group<split) {
            const unsigned position=target[short_group];
            REAL value=values[position];
            for(unsigned p=pointer[short_group];p<pointer[short_group+1];++p) {
                const unsigned u=Packed?position-unsigned(upper_delta[p]):upper[p];
                value-=values[lower[p]]*values[u];
            }
            values[position]=value;
        }
        __syncthreads();
        const unsigned long_group=split+warp;
        if(long_group<gb) {
            const unsigned position=target[long_group];
            REAL sum=REAL(0);
            for(unsigned p=pointer[long_group]+lane;p<pointer[long_group+1];p+=warpSize) {
                const unsigned u=Packed?position-unsigned(upper_delta[p]):upper[p];
                sum+=values[lower[p]]*values[u];
            }
            for(unsigned offset=warpSize/2;offset;offset/=2)
                sum+=__shfl_down_sync(~0ULL,sum,offset,warpSize);
            if(lane==0) values[position]-=sum;
        }
        __syncthreads();
    }
}

// Exact sparse-prefix panels keep values in the original CSC factor storage.
struct SparsePanelState {
    unsigned *head=nullptr,*width=nullptr,*diagonal=nullptr;
    unsigned *lower_node=nullptr,*lower_first=nullptr,*upper_node=nullptr,*upper_first=nullptr;
    unsigned *schur_node=nullptr,*schur_row=nullptr,*schur_col=nullptr,*singleton_entry=nullptr;
    unsigned *singleton_offset=nullptr,*singleton_mapped=nullptr,*singleton_fallback=nullptr;
    unsigned short *singleton_delta=nullptr;
    unsigned *micro_node=nullptr,*micro_row=nullptr,*micro_col=nullptr,*micro_fringe=nullptr;
    unsigned short *micro_delta=nullptr,*fringe_delta=nullptr;
    unsigned *fringe_offset=nullptr;
    unsigned *normalize_position=nullptr,*normalize_diagonal=nullptr;
    int *info=nullptr;
    std::vector<void*> allocations;
    std::vector<cudaGraph_t> graphs;
    std::vector<cudaGraphExec_t> execs;
    ~SparsePanelState() {
        for(auto e:execs) if(e) cudaGraphExecDestroy(e);
        for(auto g:graphs) if(g) cudaGraphDestroy(g);
        for(auto p:allocations) if(p) cudaFree(p);
        if(info) cudaFree(info);
    }
};

__global__ void sparse_panel_diagonal(REAL *val,const unsigned *diag,
        const unsigned *heads,const unsigned *widths,const unsigned *nodes,
        unsigned first,int *info) {
    const unsigned node=nodes[first+blockIdx.x],head=heads[node],size=widths[node];
    const unsigned row=threadIdx.x%16,col=threadIdx.x/16;
    __shared__ REAL a[16][16];
    __shared__ bool failed;
    if(threadIdx.x==0) failed=false;
    a[col][row]=(row<size && col<size)?val[diag[head+col]-col+row]:REAL(0);
    __syncthreads();
    for(unsigned k=0;k<size;++k) {
        if(threadIdx.x==0) {
            const REAL pivot=a[k][k];
            if(pivot==REAL(0) || !(fabs(pivot)<=REAL(1.7976931348623157e308))) {
                atomicCAS(info,0,int(head+k+1));failed=true;
            }
        }
        __syncthreads();
        if(failed) return;
        if(col==k && row>k && row<size) a[k][row]/=a[k][k];
        __syncthreads();
        if(row>k && row<size && col>k && col<size) a[col][row]-=a[k][row]*a[col][k];
        __syncthreads();
    }
    if(row<size && col<size) val[diag[head+col]-col+row]=a[col][row];
}

__global__ void sparse_panel_lower(REAL *val,const unsigned *cp,const unsigned *diag,
        const unsigned *heads,const unsigned *widths,const unsigned *nodes,
        const unsigned *starts,unsigned first) {
    const unsigned task=first+blockIdx.x,node=nodes[task],head=heads[node],size=widths[node];
    const unsigned last=head+size-1,R=cp[last+1]-diag[last]-1;
    __shared__ REAL upper[16][16];
    const unsigned a=threadIdx.x%16,b=threadIdx.x/16;
    upper[b][a]=(a<size && b<size)?val[diag[head+b]-b+a]:REAL(0);
    __syncthreads();
    const unsigned row=starts[task]+threadIdx.x;
    if(row>=R) return;
    REAL values[16];
#pragma unroll
    for(unsigned j=0;j<16;++j) values[j]=j<size?val[diag[head+j]+size-j+row]:REAL(0);
#pragma unroll
    for(unsigned k=0;k<16;++k) if(k<size) {
        const REAL lower=values[k]/upper[k][k];values[k]=lower;
#pragma unroll
        for(unsigned j=0;j<16;++j) if(j>k && j<size) values[j]-=lower*upper[j][k];
    }
#pragma unroll
    for(unsigned j=0;j<16;++j) if(j<size) val[diag[head+j]+size-j+row]=values[j];
}

__global__ void sparse_panel_upper(REAL *val,const unsigned *diag,const unsigned *rp,
        const unsigned *rd,const unsigned *pivot,const unsigned *heads,const unsigned *widths,
        const unsigned *nodes,const unsigned *starts,unsigned first) {
    const unsigned task=first+blockIdx.x,node=nodes[task],head=heads[node],size=widths[node];
    const unsigned last=head+size-1,C=rp[last+1]-rd[last]-1;
    __shared__ REAL lower[16][16];
    const unsigned a=threadIdx.x%16,b=threadIdx.x/16;
    lower[b][a]=(a<size && b<size)?val[diag[head+b]-b+a]:REAL(0);
    __syncthreads();
    const unsigned col=starts[task]+threadIdx.x;
    if(col>=C) return;
    const unsigned base=pivot[rd[last]+1+col]-size+1;
    REAL values[16];
#pragma unroll
    for(unsigned j=0;j<16;++j) values[j]=j<size?val[base+j]:REAL(0);
#pragma unroll
    for(unsigned k=0;k<16;++k) if(k<size) {
        const REAL upper=values[k];
#pragma unroll
        for(unsigned j=0;j<16;++j) if(j>k && j<size) values[j]-=upper*lower[k][j];
    }
    // The first row is unchanged; avoid redundant writes to coefficient readers.
#pragma unroll
    for(unsigned j=1;j<16;++j) if(j<size) val[base+j]=values[j];
}

__global__ void sparse_panel_schur(REAL *val,const unsigned *cp,const unsigned *ri,
        const unsigned *diag,const unsigned *rp,const unsigned *ci,const unsigned *rd,
        const unsigned *pivot,const unsigned *heads,const unsigned *widths,
        const unsigned *nodes,const unsigned *starts_r,const unsigned *starts_c,
        unsigned first,int *info) {
    const unsigned task=first+blockIdx.x,node=nodes[task],head=heads[node],size=widths[node];
    const unsigned last=head+size-1,R=cp[last+1]-diag[last]-1,C=rp[last+1]-rd[last]-1;
    const unsigned x=threadIdx.x,y=threadIdx.y;
    const unsigned row_index=starts_r[task]+x,col_index=starts_c[task]+y;
    __shared__ REAL lower[16][16],upper[16][16];
    lower[y][x]=(row_index<R && y<size)?val[diag[head+y]+size-y+row_index]:REAL(0);
    upper[y][x]=(col_index<C && x<size)?val[pivot[rd[last]+1+col_index]-size+1+x]:REAL(0);
    __syncthreads();
    if(row_index>=R || col_index>=C) return;
    const unsigned row=ri[diag[last]+1+row_index],entry=rd[last]+1+col_index,target=ci[entry];
    unsigned lo=pivot[entry]+1,hi=cp[target+1];
    while(lo<hi) {
        const unsigned mid=lo+(hi-lo)/2;
        if(ri[mid]<row) lo=mid+1;else hi=mid;
    }
    if(lo==cp[target+1] || ri[lo]!=row) {atomicCAS(info,0,int(head+1));return;}
    REAL sum=REAL(0);
    for(unsigned k=0;k<size;++k) sum+=lower[k][x]*upper[y][k];
    // Independent macro nodes can share a target: one atomic per panel-cell.
    atomicAdd(&val[lo],-sum);
}

// Four register outputs per thread share the same lower operand in each k step.
template<unsigned Width>
__device__ __forceinline__ void sparse_panel_exact_products(
        const REAL lower[16][32],const REAL upper[32][16],unsigned row,unsigned group,
        REAL &s0,REAL &s1,REAL &s2,REAL &s3) {
    // Width is the existing symbolic panel size, not a tunable tile size.
    // Preserve the original ascending-k recurrence for each output.
    #pragma unroll
    for(unsigned j=0;j<Width;++j) {
        const REAL l=lower[j][row];
        s0+=l*upper[group][j];s1+=l*upper[group+8][j];
        s2+=l*upper[group+16][j];s3+=l*upper[group+24][j];
    }
}

__global__ void sparse_panel_microtiles(REAL *val,const unsigned *diag,const unsigned *rd,
        const unsigned *pivot,const unsigned *heads,const unsigned *widths,const unsigned *nodes,
        const unsigned *starts_r,const unsigned *starts_c,const unsigned short *deltas,unsigned first) {
    const unsigned task=first+blockIdx.x,node=nodes[task],head=heads[node],size=widths[node],last=head+size-1;
    const unsigned r=starts_r[task],c=starts_c[task],row=threadIdx.x%32,group=threadIdx.x/32;
    __shared__ REAL lower[16][32],upper[32][16];
    const unsigned k0=threadIdx.x/32,k1=k0+8;
    lower[k0][row]=k0<size?val[diag[head+k0]+size-k0+r+row]:REAL(0);
    lower[k1][row]=k1<size?val[diag[head+k1]+size-k1+r+row]:REAL(0);
    const unsigned k=threadIdx.x%16,col=threadIdx.x/16,e0=rd[last]+1+c+col;
    upper[col][k]=k<size?val[pivot[e0]-size+1+k]:REAL(0);
    upper[col+16][k]=k<size?val[pivot[e0+16]-size+1+k]:REAL(0);
    __syncthreads();
    REAL s0=REAL(0),s1=REAL(0),s2=REAL(0),s3=REAL(0);
    // All threads in this CTA read the same immutable node width. Shared
    // storage and the preparation barrier above are common to every shape.
    switch(size) {
        case 2: sparse_panel_exact_products<2>(lower,upper,row,group,s0,s1,s2,s3);break;
        case 3: sparse_panel_exact_products<3>(lower,upper,row,group,s0,s1,s2,s3);break;
        case 4: sparse_panel_exact_products<4>(lower,upper,row,group,s0,s1,s2,s3);break;
        case 5: sparse_panel_exact_products<5>(lower,upper,row,group,s0,s1,s2,s3);break;
        case 6: sparse_panel_exact_products<6>(lower,upper,row,group,s0,s1,s2,s3);break;
        case 7: sparse_panel_exact_products<7>(lower,upper,row,group,s0,s1,s2,s3);break;
        case 8: sparse_panel_exact_products<8>(lower,upper,row,group,s0,s1,s2,s3);break;
        case 9: sparse_panel_exact_products<9>(lower,upper,row,group,s0,s1,s2,s3);break;
        case 10: sparse_panel_exact_products<10>(lower,upper,row,group,s0,s1,s2,s3);break;
        case 11: sparse_panel_exact_products<11>(lower,upper,row,group,s0,s1,s2,s3);break;
        case 12: sparse_panel_exact_products<12>(lower,upper,row,group,s0,s1,s2,s3);break;
        case 13: sparse_panel_exact_products<13>(lower,upper,row,group,s0,s1,s2,s3);break;
        case 14: sparse_panel_exact_products<14>(lower,upper,row,group,s0,s1,s2,s3);break;
        case 15: sparse_panel_exact_products<15>(lower,upper,row,group,s0,s1,s2,s3);break;
        case 16: sparse_panel_exact_products<16>(lower,upper,row,group,s0,s1,s2,s3);break;
        default:
            for(unsigned j=0;j<size;++j) {
                const REAL l=lower[j][row];
                s0+=l*upper[group][j];s1+=l*upper[group+8][j];
                s2+=l*upper[group+16][j];s3+=l*upper[group+24][j];
            }
    }
    const unsigned entry=rd[last]+1+c+group;
    const size_t offset=size_t(task)*1024+group*32+row;
    atomicAdd(&val[pivot[entry]+1+unsigned(deltas[offset])],-s0);
    atomicAdd(&val[pivot[entry+8]+1+unsigned(deltas[offset+8*32])],-s1);
    atomicAdd(&val[pivot[entry+16]+1+unsigned(deltas[offset+16*32])],-s2);
    atomicAdd(&val[pivot[entry+24]+1+unsigned(deltas[offset+24*32])],-s3);
}

__global__ void sparse_panel_queued_fringe(REAL *val,const unsigned *cp,const unsigned *ri,
        const unsigned *diag,const unsigned *rp,const unsigned *ci,const unsigned *rd,
        const unsigned *pivot,const unsigned *heads,const unsigned *widths,
        const unsigned *nodes,const unsigned *starts_r,const unsigned *starts_c,
        const unsigned *tasks,unsigned first,int *info) {
    const unsigned task=tasks[first+blockIdx.x],node=nodes[task],head=heads[node],size=widths[node];
    const unsigned last=head+size-1,R=cp[last+1]-diag[last]-1,C=rp[last+1]-rd[last]-1;
    const unsigned x=threadIdx.x,y=threadIdx.y;
    const unsigned row_index=starts_r[task]+x,col_index=starts_c[task]+y;
    __shared__ REAL lower[16][16],upper[16][16];
    lower[y][x]=(row_index<R && y<size)?val[diag[head+y]+size-y+row_index]:REAL(0);
    upper[y][x]=(col_index<C && x<size)?val[pivot[rd[last]+1+col_index]-size+1+x]:REAL(0);
    __syncthreads();
    if(row_index>=R || col_index>=C) return;
    const unsigned row=ri[diag[last]+1+row_index],entry=rd[last]+1+col_index,target=ci[entry];
    unsigned lo=pivot[entry]+1,hi=cp[target+1];
    while(lo<hi) {
        const unsigned mid=lo+(hi-lo)/2;
        if(ri[mid]<row) lo=mid+1;else hi=mid;
    }
    if(lo==cp[target+1] || ri[lo]!=row) {atomicCAS(info,0,int(head+1));return;}
    REAL sum=REAL(0);
    for(unsigned j=0;j<size;++j) sum+=lower[j][x]*upper[y][j];
    atomicAdd(&val[lo],-sum);
}

__global__ void sparse_panel_singletons(REAL *val,const unsigned *cp,const unsigned *ri,
        const unsigned *diag,const unsigned *ci,const unsigned *pivot,const unsigned *entries,
        unsigned first,unsigned end,int *info) {
    const unsigned task=first+(blockIdx.x*blockDim.x+threadIdx.x)/warpSize;
    if(task>=end) return;
    const unsigned e=entries[task],pp=pivot[e],source=ri[pp],target=ci[e];
    const unsigned lane=threadIdx.x%warpSize;
    const REAL coefficient=val[pp];
    for(unsigned p=diag[source]+1+lane;p<cp[source+1];p+=warpSize) {
        const unsigned row=ri[p];unsigned lo=pp+1,hi=cp[target+1];
        while(lo<hi) {const unsigned mid=lo+(hi-lo)/2;if(ri[mid]<row)lo=mid+1;else hi=mid;}
        if(lo==cp[target+1] || ri[lo]!=row) {atomicCAS(info,0,int(source+1));continue;}
        atomicAdd(&val[lo],-val[p]*coefficient);
    }
}

// Every task here has a complete, CPU-verified structural target stream.
__global__ void sparse_singletons_compiled(REAL *val,const unsigned *cp,const unsigned *ri,
        const unsigned *diag,const unsigned *pivot,const unsigned *entries,const unsigned *tasks,
        const unsigned *offsets,const unsigned short *deltas,unsigned first,unsigned end) {
    const unsigned task=first+(blockIdx.x*blockDim.x+threadIdx.x)/warpSize;
    if(task>=end) return;
    const unsigned original=tasks[task],e=entries[original],pp=pivot[e],source=ri[pp];
    const unsigned begin=diag[source]+1,count=cp[source+1]-begin,offset=offsets[original];
    const REAL coefficient=val[pp];
    for(unsigned rank=threadIdx.x%warpSize;rank<count;rank+=warpSize)
        atomicAdd(&val[pp+1+unsigned(deltas[offset+rank])],-val[begin+rank]*coefficient);
}

// Keep search resources out of the complete-operand kernel.
__global__ void sparse_singletons_queued_fallback(REAL *val,const unsigned *cp,const unsigned *ri,
        const unsigned *diag,const unsigned *ci,const unsigned *pivot,const unsigned *entries,
        const unsigned *tasks,unsigned first,unsigned end,int *info) {
    const unsigned task=first+(blockIdx.x*blockDim.x+threadIdx.x)/warpSize;
    if(task>=end) return;
    const unsigned e=entries[tasks[task]],pp=pivot[e],source=ri[pp],target=ci[e];
    const unsigned lane=threadIdx.x%warpSize;
    const REAL coefficient=val[pp];
    for(unsigned p=diag[source]+1+lane;p<cp[source+1];p+=warpSize) {
        const unsigned row=ri[p];unsigned lo=pp+1,hi=cp[target+1];
        while(lo<hi) {const unsigned mid=lo+(hi-lo)/2;if(ri[mid]<row)lo=mid+1;else hi=mid;}
        if(lo==cp[target+1] || ri[lo]!=row) {atomicCAS(info,0,int(source+1));continue;}
        atomicAdd(&val[lo],-val[p]*coefficient);
    }
}

__device__ __forceinline__ bool sparse_local_factor(REAL a[16][16],unsigned head,
        unsigned size,int *info,int *failure) {
    const unsigned tid=threadIdx.x+blockDim.x*threadIdx.y,row=tid%16,col=tid/16;
    if(tid==0) *failure=0;
    __syncthreads();
    if(size==1) return true;
    for(unsigned k=0;k<size;++k) {
        if(tid==0) {
            const REAL pivot=a[k][k];
            if(pivot==REAL(0) || !(fabs(pivot)<=REAL(1.7976931348623157e308))) {
                atomicCAS(info,0,int(head+k+1));*failure=1;
            }
        }
        __syncthreads();
        if(*failure) return false;
        if(col==k && row>k && row<size) a[k][row]/=a[k][k];
        __syncthreads();
        if(row>k && row<size && col>k && col<size) a[col][row]-=a[k][row]*a[col][k];
        __syncthreads();
    }
    return true;
}

// Only multi-node RHS tasks enter here. Global A11 stays original until commit.
__global__ void sparse_panel_prepare_local(REAL *val,const unsigned *cp,const unsigned *diag,
        const unsigned *rp,const unsigned *rd,const unsigned *pivot,
        const unsigned *heads,const unsigned *widths,
        const unsigned *lower_nodes,const unsigned *lower_starts,
        const unsigned *upper_nodes,const unsigned *upper_starts,
        unsigned lower_first,unsigned lower_count,unsigned upper_first,int *info) {
    const bool is_lower=blockIdx.x<lower_count;
    const unsigned task=is_lower?lower_first+blockIdx.x:upper_first+blockIdx.x-lower_count;
    const unsigned node=is_lower?lower_nodes[task]:upper_nodes[task];
    const unsigned first=is_lower?lower_starts[task]:upper_starts[task];
    const unsigned head=heads[node],size=widths[node],last=head+size-1;
    __shared__ REAL a[16][16];__shared__ int failure;
    const unsigned row=threadIdx.x%16,col=threadIdx.x/16;
    a[col][row]=(row<size && col<size)?val[diag[head+col]-col+row]:REAL(0);
    if(!sparse_local_factor(a,head,size,info,&failure)) return;
    const unsigned index=first+threadIdx.x;
    if(is_lower) {
        const unsigned R=cp[last+1]-diag[last]-1;
        if(index>=R) return;
        REAL values[16];
#pragma unroll
        for(unsigned j=0;j<16;++j) values[j]=j<size?val[diag[head+j]+size-j+index]:REAL(0);
#pragma unroll
        for(unsigned k=0;k<16;++k) if(k<size) {
            const REAL lower=values[k]/a[k][k];values[k]=lower;
#pragma unroll
            for(unsigned j=0;j<16;++j) if(j>k && j<size) values[j]-=lower*a[j][k];
        }
#pragma unroll
        for(unsigned j=0;j<16;++j) if(j<size) val[diag[head+j]+size-j+index]=values[j];
    } else {
        const unsigned C=rp[last+1]-rd[last]-1;
        if(index>=C) return;
        const unsigned base=pivot[rd[last]+1+index]-size+1;
        REAL values[16];
#pragma unroll
        for(unsigned j=0;j<16;++j) values[j]=j<size?val[base+j]:REAL(0);
#pragma unroll
        for(unsigned k=0;k<16;++k) if(k<size) {
            const REAL upper=values[k];
#pragma unroll
            for(unsigned j=0;j<16;++j) if(j>k && j<size) values[j]-=upper*a[k][j];
        }
#pragma unroll
        for(unsigned j=1;j<16;++j) if(j<size) val[base+j]=values[j];
    }
}

// Three independent CTA roles after prepare: LU commit, old fringe update, flat norm.
__global__ void sparse_panel_commit_fringe_norm(REAL *val,const unsigned *cp,const unsigned *ri,
        const unsigned *diag,const unsigned *rp,const unsigned *ci,const unsigned *rd,
        const unsigned *pivot,const unsigned *heads,const unsigned *widths,
        const unsigned *diagonal_nodes,unsigned diagonal_first,unsigned diagonal_count,
        const unsigned *nodes,const unsigned *starts_r,const unsigned *starts_c,
        const unsigned *fringe_tasks,unsigned fringe_first,unsigned fringe_count,
        const unsigned *norm_position,const unsigned *norm_diagonal,unsigned norm_first,unsigned norm_end,
        const unsigned *fringe_offset,const unsigned short *fringe_delta,int *info) {
    const unsigned tid=threadIdx.x;
    if(blockIdx.x>=diagonal_count+fringe_count) {
        const unsigned p=norm_first+(blockIdx.x-diagonal_count-fringe_count)*blockDim.x+tid;
        if(p<norm_end) val[norm_position[p]]/=val[norm_diagonal[p]];
        return; // CTA-uniform role: no norm thread reaches a shared barrier.
    }
    __shared__ REAL lower[16][16],upper[16][16];__shared__ int failure;
    const unsigned x=tid%16,y=tid/16;
    if(blockIdx.x<diagonal_count) {
        const unsigned node=diagonal_nodes[diagonal_first+blockIdx.x],head=heads[node],size=widths[node];
        upper[y][x]=(x<size && y<size)?val[diag[head+y]-y+x]:REAL(0);
        if(!sparse_local_factor(upper,head,size,info,&failure)) return;
        if(x<size && y<size) val[diag[head+y]-y+x]=upper[y][x];
        return;
    }
    const unsigned q=fringe_first+blockIdx.x-diagonal_count;
    const unsigned task=fringe_tasks?fringe_tasks[q]:q;
    const unsigned node=nodes[task],head=heads[node],size=widths[node],last=head+size-1;
    const unsigned R=cp[last+1]-diag[last]-1,C=rp[last+1]-rd[last]-1;
    const unsigned row_index=starts_r[task]+x,col_index=starts_c[task]+y;
    lower[y][x]=(row_index<R && y<size)?val[diag[head+y]+size-y+row_index]:REAL(0);
    upper[y][x]=(col_index<C && x<size)?val[pivot[rd[last]+1+col_index]-size+1+x]:REAL(0);
    __syncthreads();
    if(row_index>=R || col_index>=C) return;
    const unsigned row=ri[diag[last]+1+row_index],entry=rd[last]+1+col_index,target=ci[entry];
    const unsigned address=fringe_offset?fringe_offset[q]:~0u;
    unsigned lo;
    if(address!=~0u) lo=pivot[entry]+1+unsigned(fringe_delta[address+16*y+x]);
    else {
        lo=pivot[entry]+1;unsigned hi=cp[target+1];
        while(lo<hi) {const unsigned mid=lo+(hi-lo)/2;if(ri[mid]<row)lo=mid+1;else hi=mid;}
        if(lo==cp[target+1] || ri[lo]!=row) {atomicCAS(info,0,int(head+1));return;}
    }
    REAL sum=REAL(0);
    for(unsigned k=0;k<size;++k) sum+=lower[k][x]*upper[y][k];
    atomicAdd(&val[lo],-sum);
}

__global__ void sparse_singletons_compiled_short(REAL *val,const unsigned *cp,const unsigned *ri,
        const unsigned *diag,const unsigned *pivot,const unsigned *entries,const unsigned *tasks,
        const unsigned *offsets,const unsigned short *deltas,unsigned first,unsigned end) {
    const unsigned task=first+(blockIdx.x*blockDim.x+threadIdx.x)/16;
    if(task>=end) return;
    const unsigned original=tasks[task],e=entries[original],pp=pivot[e],source=ri[pp];
    const unsigned begin=diag[source]+1,count=cp[source+1]-begin,rank=threadIdx.x%16;
    if(rank<count) atomicAdd(&val[pp+1+unsigned(deltas[offsets[original]+rank])],-val[begin+rank]*val[pp]);
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



// New working-set representation: the complete diagonal A11 block stays
// in shared memory for every original ordered FP64 pivot/update.
__global__ void dense_diagonal_shared(REAL *a,unsigned n,unsigned first,
        unsigned end,int *info) {
    if(*info) return;
    __shared__ REAL tile[256],pivot;
    const unsigned p=threadIdx.x,row=p%16,col=p/16,size=end-first;
    const bool active=row<size && col<size;
    tile[p]=active?a[first+row+size_t(n)*(first+col)]:REAL(0);
    __syncthreads();
    for(unsigned k=0;k<size;++k) {
        if(p==0) {
            pivot=tile[k+16*k];
            if(pivot==REAL(0)) atomicCAS(info,0,int(first+k+1));
        }
        __syncthreads();
        if(*info) return; // uniform CTA after pivot publication
        if(col==k && row>k && row<size) tile[p]/=pivot;
        __syncthreads();
        if(active && row>k && col>k) tile[p]-=tile[row+16*k]*tile[k+16*col];
        __syncthreads();
    }
    if(active) a[first+row+size_t(n)*(first+col)]=tile[p];
}

// Parallel register-resident L21 solve was already explored in c97.
// Reuse that mathematical decomposition; it is not counted as new here.
__global__ void dense_panel_lower_register(REAL *a,unsigned n,unsigned first,
        unsigned end,const int *info) {
    if(*info) return;
    const unsigned row=end+blockIdx.x*blockDim.x+threadIdx.x;
    if(row>=n) return;
    const unsigned size=end-first;
    REAL values[16];
    #pragma unroll
    for(unsigned j=0;j<16;++j)
        values[j]=j<size?a[row+size_t(n)*(first+j)]:REAL(0);
    #pragma unroll
    for(unsigned k=0;k<16;++k) {
        if(k<size) {
            const REAL lower=values[k]/a[first+k+size_t(n)*(first+k)];
            values[k]=lower;
            #pragma unroll
            for(unsigned j=0;j<16;++j)
                if(j>k && j<size) values[j]-=lower*a[first+k+size_t(n)*(first+j)];
        }
    }
    #pragma unroll
    for(unsigned j=0;j<16;++j)
        if(j<size) a[row+size_t(n)*(first+j)]=values[j];
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
        const int inLevPos, unsigned deferred_cut)
{
    // A whole CTA owns one source; return before entering RL_body barriers.
    if(unsigned(level_idx_dev[levelHead+inLevPos+blockIdx.x])>=deferred_cut) return;
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
        unsigned queue_first, unsigned queue_end, unsigned deferred_cut) {
    const unsigned owner = blockIdx.x * (blockDim.x / warpSize) + threadIdx.x / warpSize;
    const unsigned position = queue_first + owner;
    if (position >= queue_end) return;
    const unsigned col = level_idx[source_queue[position]];
    if(col>=deferred_cut) return;
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
        const unsigned *source_queue, unsigned queue_first, unsigned deferred_cut) {
    const unsigned slot = source_queue ? source_queue[queue_first + blockIdx.x] - source_head : blockIdx.x;
    const unsigned col = level_idx[source_head + slot];
    if(col>=deferred_cut) return; // uniform CTA return before its barrier
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

// Ordinary writeback is safe only when symbolic analysis proves one source
// owns this entire target column in the original level. Other writers stay atomic.
__device__ __forceinline__ void sparse_task_writeback(REAL *target, REAL delta,
                                                     bool exclusive) {
    if(exclusive) *target+=delta;
    else atomicAdd(target,delta);
}

__global__ void RL_update_short_tasks(REAL *val,
        const unsigned *pivot, const unsigned *task_entry, const unsigned *task_index,
        const unsigned *pair_begin, const unsigned *pair_end,
        const unsigned *pair_source, const unsigned *pair_target,
        const unsigned *task_source,
        unsigned head, unsigned end, const unsigned *row_index, unsigned deferred_cut) {
    const unsigned logical = head + blockIdx.x * (blockDim.x / 16) + threadIdx.x / 16;
    if (logical >= end) return; // no warp collectives in this kernel
    const unsigned task = task_index[logical];
    if(row_index[pivot[task_entry[task]]]>=deferred_cut) return;
    const unsigned p = pair_begin[task] + threadIdx.x % 16;
    if (p < pair_end[task]) {
        const REAL coefficient = val[pivot[task_entry[task]]];
        sparse_task_writeback(&val[pair_target[p]], -val[pair_source[p]] * coefficient,
                              (task_source[task]&0x80000000u)!=0);
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
        bool skip_short_tasks, unsigned deferred_cut) {
    const unsigned tasks_per_block = blockDim.x / warpSize;
    const unsigned task = task_head + blockIdx.x * tasks_per_block + threadIdx.x / warpSize;
    if (task >= task_end) return; // whole native warp exits together
    if (skip_short_tasks && pair_begin[task] != ~0u &&
        pair_end[task] - pair_begin[task] <= 16) return;
    const unsigned encoded_source = task_source[task];
    const unsigned source_pos = encoded_source&0x7fffffffu;
    const bool exclusive = (encoded_source&0x80000000u)!=0;
    const unsigned bid = source_pos - source_head;
    const unsigned currentCol = level_idx_dev[source_pos];
    if(currentCol>=deferred_cut) return; // whole native warp
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
                sparse_task_writeback(&val_dev[pair_target[p]], -val_dev[pair_source[p]] * coefficient, exclusive);
            continue;
        }
        if (task_begin[task] != ~0u) {
            const REAL coefficient = val_dev[csr_pivot_pos_dev[entry]];
            for (unsigned pos = task_begin[task] + lane; pos < task_end_pos[task]; pos += warpSize) {
                const unsigned row = sym_r_idx_dev[pos];
                sparse_task_writeback(&val_dev[pos], -scratch_value(tmpMem, scratch_tag, size_t(bid) * n + row, currentCol) * coefficient, exclusive);
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
                    sparse_task_writeback(&val_dev[pos], -scratch_value(tmpMem, scratch_tag, size_t(bid) * n + row, currentCol) * short_coefficient, exclusive);
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
                    sparse_task_writeback(&val_dev[lo], -scratch_value(tmpMem, scratch_tag, size_t(bid) * n + row, currentCol) * coefficient, exclusive);
                    ++lo;
                }
            }
        } else {
            for (unsigned pos = first_lower + lane; pos < end; pos += warpSize) {
                const unsigned row = sym_r_idx_dev[pos];
                sparse_task_writeback(&val_dev[pos], -scratch_value(tmpMem, scratch_tag, size_t(bid) * n + row, currentCol) * coefficient, exclusive);
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
    std::shared_ptr<SparsePanelState> panel_state;
    const bool panel_active=!PERTURB && deviceProp.warpSize==64 && A_sym.macro_ready;
    const bool destination_owned = !PERTURB && A_sym.destination_ready;
    const bool destination_root = destination_owned && A_sym.destination_root_only;
    const bool destination_whole = destination_owned && !A_sym.destination_root_only;
    const bool destination_flat = destination_owned && A_sym.destination_flat_normalization;
    const bool destination_packed = destination_owned && A_sym.destination_upper_packed;
    const bool destination_cta = destination_whole && destination_flat && deviceProp.warpSize==64 &&
        A_sym.destination_normalize_ptr.size()==num_lev+1 && A_sym.destination_level_ptr.size()==num_lev+1 &&
        (A_sym.destination_heavy_level_ptr.empty() || A_sym.destination_heavy_level_ptr.size()==num_lev);
    const bool deferred_active = !PERTURB && A_sym.dense_deferred;
    const unsigned deferred_cut = deferred_active ? A_sym.dense_deferred_cut : n;
    unsigned deferred_skipped_levels=0;
    if(deferred_active) out << "Deferred Schur execution active: sparse prefix " << deferred_cut
        << ", dense suffix " << n-deferred_cut << endl;
    const unsigned dense_size = (PERTURB || destination_whole) ? 0u : A_sym.dense_columns.size();
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
        panel_state.reset();
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
    if(panel_active) {
        panel_state=std::make_shared<SparsePanelState>();
        struct PanelArray {unsigned **destination;const std::vector<unsigned> *source;};
        const PanelArray arrays[]={
            {&panel_state->head,&A_sym.macro_head},
            {&panel_state->width,&A_sym.macro_width},
            {&panel_state->diagonal,&A_sym.macro_diagonal},
            {&panel_state->lower_node,&A_sym.macro_lower_node},
            {&panel_state->lower_first,&A_sym.macro_lower_first},
            {&panel_state->upper_node,&A_sym.macro_upper_node},
            {&panel_state->upper_first,&A_sym.macro_upper_first},
            {&panel_state->schur_node,&A_sym.macro_schur_node},
            {&panel_state->schur_row,&A_sym.macro_schur_row},
            {&panel_state->schur_col,&A_sym.macro_schur_col},
            {&panel_state->singleton_entry,&A_sym.macro_singleton_entry},
            {&panel_state->singleton_offset,&A_sym.macro_singleton_offset},
            {&panel_state->singleton_mapped,&A_sym.macro_singleton_mapped},
            {&panel_state->singleton_fallback,&A_sym.macro_singleton_fallback},
            {&panel_state->micro_node,&A_sym.macro_micro_node},
            {&panel_state->micro_row,&A_sym.macro_micro_row},
            {&panel_state->micro_col,&A_sym.macro_micro_col},
            {&panel_state->micro_fringe,&A_sym.macro_micro_fringe},
            {&panel_state->fringe_offset,&A_sym.macro_fringe_offset},
            {&panel_state->normalize_position,&A_sym.macro_normalize_position},
            {&panel_state->normalize_diagonal,&A_sym.macro_normalize_diagonal}
        };
        for(const auto &array:arrays) if(!array.source->empty()) {
            CUDA_RETURN_ON_ERR(cudaMalloc((void**)array.destination,array.source->size()*sizeof(unsigned)),"sparse panel allocation");
            panel_state->allocations.push_back(*array.destination);
        }
        CUDA_RETURN_ON_ERR(cudaMalloc((void**)&panel_state->info,sizeof(int)),"sparse panel info allocation");
        if(!A_sym.macro_singleton_delta.empty()) {
            CUDA_RETURN_ON_ERR(cudaMalloc((void**)&panel_state->singleton_delta,A_sym.macro_singleton_delta.size()*sizeof(unsigned short)),"compact singleton operand allocation");
            panel_state->allocations.push_back(panel_state->singleton_delta);
        }
        if(!A_sym.macro_micro_delta.empty()) {
            CUDA_RETURN_ON_ERR(cudaMalloc((void**)&panel_state->micro_delta,A_sym.macro_micro_delta.size()*sizeof(unsigned short)),"sparse microtile operand allocation");
            panel_state->allocations.push_back(panel_state->micro_delta);
        }
        if(!A_sym.macro_fringe_delta.empty()) {
            CUDA_RETURN_ON_ERR(cudaMalloc((void**)&panel_state->fringe_delta,A_sym.macro_fringe_delta.size()*sizeof(unsigned short)),"sparse fringe operand allocation");
            panel_state->allocations.push_back(panel_state->fringe_delta);
        }
    }
    if (destination_owned) {
        destination_state = std::make_shared<DestinationGroupState>();
        CUDA_RETURN_ON_ERR(cudaMalloc((void**)&destination_state->lower, A_sym.destination_lower.size() * sizeof(unsigned)), "destination lower allocation");
        if(destination_packed)
            CUDA_RETURN_ON_ERR(cudaMalloc((void**)&destination_state->upper_delta,A_sym.destination_upper_delta.size()*sizeof(unsigned short)),"destination relative upper allocation");
        else CUDA_RETURN_ON_ERR(cudaMalloc((void**)&destination_state->upper, A_sym.destination_upper.size() * sizeof(unsigned)), "destination upper allocation");
        CUDA_RETURN_ON_ERR(cudaMalloc((void**)&destination_state->pointer, A_sym.destination_group_ptr.size() * sizeof(unsigned)), "destination pointer allocation");
        CUDA_RETURN_ON_ERR(cudaMalloc((void**)&destination_state->target, A_sym.destination_target.size() * sizeof(unsigned)), "destination target allocation");
        if (destination_flat) {
            CUDA_RETURN_ON_ERR(cudaMalloc((void**)&destination_state->normalize_lower, A_sym.destination_normalize_lower.size()*sizeof(unsigned)), "root lower allocation");
            CUDA_RETURN_ON_ERR(cudaMalloc((void**)&destination_state->normalize_diagonal, A_sym.destination_normalize_diagonal.size()*sizeof(unsigned)), "root diagonal allocation");
        }
        if(destination_cta) {
            CUDA_RETURN_ON_ERR(cudaMalloc((void**)&destination_state->normalize_level,A_sym.destination_normalize_ptr.size()*sizeof(unsigned)),"gather normalization prefix allocation");
            CUDA_RETURN_ON_ERR(cudaMalloc((void**)&destination_state->group_level,A_sym.destination_level_ptr.size()*sizeof(unsigned)),"gather group prefix allocation");
            if(!A_sym.destination_heavy_level_ptr.empty())
                CUDA_RETURN_ON_ERR(cudaMalloc((void**)&destination_state->heavy_level,A_sym.destination_heavy_level_ptr.size()*sizeof(unsigned)),"gather heavy prefix allocation");
        }
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
    if(panel_state) {
        if(panel_state->head) CUDA_RETURN_ON_ERR(cudaMemcpy(panel_state->head,A_sym.macro_head.data(),A_sym.macro_head.size()*sizeof(unsigned),cudaMemcpyHostToDevice),"sparse panel upload");
        if(panel_state->width) CUDA_RETURN_ON_ERR(cudaMemcpy(panel_state->width,A_sym.macro_width.data(),A_sym.macro_width.size()*sizeof(unsigned),cudaMemcpyHostToDevice),"sparse panel upload");
        if(panel_state->diagonal) CUDA_RETURN_ON_ERR(cudaMemcpy(panel_state->diagonal,A_sym.macro_diagonal.data(),A_sym.macro_diagonal.size()*sizeof(unsigned),cudaMemcpyHostToDevice),"sparse panel upload");
        if(panel_state->lower_node) CUDA_RETURN_ON_ERR(cudaMemcpy(panel_state->lower_node,A_sym.macro_lower_node.data(),A_sym.macro_lower_node.size()*sizeof(unsigned),cudaMemcpyHostToDevice),"sparse panel upload");
        if(panel_state->lower_first) CUDA_RETURN_ON_ERR(cudaMemcpy(panel_state->lower_first,A_sym.macro_lower_first.data(),A_sym.macro_lower_first.size()*sizeof(unsigned),cudaMemcpyHostToDevice),"sparse panel upload");
        if(panel_state->upper_node) CUDA_RETURN_ON_ERR(cudaMemcpy(panel_state->upper_node,A_sym.macro_upper_node.data(),A_sym.macro_upper_node.size()*sizeof(unsigned),cudaMemcpyHostToDevice),"sparse panel upload");
        if(panel_state->upper_first) CUDA_RETURN_ON_ERR(cudaMemcpy(panel_state->upper_first,A_sym.macro_upper_first.data(),A_sym.macro_upper_first.size()*sizeof(unsigned),cudaMemcpyHostToDevice),"sparse panel upload");
        if(panel_state->schur_node) CUDA_RETURN_ON_ERR(cudaMemcpy(panel_state->schur_node,A_sym.macro_schur_node.data(),A_sym.macro_schur_node.size()*sizeof(unsigned),cudaMemcpyHostToDevice),"sparse panel upload");
        if(panel_state->schur_row) CUDA_RETURN_ON_ERR(cudaMemcpy(panel_state->schur_row,A_sym.macro_schur_row.data(),A_sym.macro_schur_row.size()*sizeof(unsigned),cudaMemcpyHostToDevice),"sparse panel upload");
        if(panel_state->schur_col) CUDA_RETURN_ON_ERR(cudaMemcpy(panel_state->schur_col,A_sym.macro_schur_col.data(),A_sym.macro_schur_col.size()*sizeof(unsigned),cudaMemcpyHostToDevice),"sparse panel upload");
        if(panel_state->singleton_entry) CUDA_RETURN_ON_ERR(cudaMemcpy(panel_state->singleton_entry,A_sym.macro_singleton_entry.data(),A_sym.macro_singleton_entry.size()*sizeof(unsigned),cudaMemcpyHostToDevice),"sparse panel upload");
        if(panel_state->singleton_offset) CUDA_RETURN_ON_ERR(cudaMemcpy(panel_state->singleton_offset,A_sym.macro_singleton_offset.data(),A_sym.macro_singleton_offset.size()*sizeof(unsigned),cudaMemcpyHostToDevice),"compact singleton header upload");
        if(panel_state->singleton_mapped) CUDA_RETURN_ON_ERR(cudaMemcpy(panel_state->singleton_mapped,A_sym.macro_singleton_mapped.data(),A_sym.macro_singleton_mapped.size()*sizeof(unsigned),cudaMemcpyHostToDevice),"compact singleton mapped queue upload");
        if(panel_state->singleton_fallback) CUDA_RETURN_ON_ERR(cudaMemcpy(panel_state->singleton_fallback,A_sym.macro_singleton_fallback.data(),A_sym.macro_singleton_fallback.size()*sizeof(unsigned),cudaMemcpyHostToDevice),"compact singleton fallback queue upload");
        if(panel_state->singleton_delta) CUDA_RETURN_ON_ERR(cudaMemcpy(panel_state->singleton_delta,A_sym.macro_singleton_delta.data(),A_sym.macro_singleton_delta.size()*sizeof(unsigned short),cudaMemcpyHostToDevice),"compact singleton operands upload");
        if(panel_state->micro_node) CUDA_RETURN_ON_ERR(cudaMemcpy(panel_state->micro_node,A_sym.macro_micro_node.data(),A_sym.macro_micro_node.size()*sizeof(unsigned),cudaMemcpyHostToDevice),"sparse microtile nodes upload");
        if(panel_state->micro_row) CUDA_RETURN_ON_ERR(cudaMemcpy(panel_state->micro_row,A_sym.macro_micro_row.data(),A_sym.macro_micro_row.size()*sizeof(unsigned),cudaMemcpyHostToDevice),"sparse microtile rows upload");
        if(panel_state->micro_col) CUDA_RETURN_ON_ERR(cudaMemcpy(panel_state->micro_col,A_sym.macro_micro_col.data(),A_sym.macro_micro_col.size()*sizeof(unsigned),cudaMemcpyHostToDevice),"sparse microtile columns upload");
        if(panel_state->micro_fringe) CUDA_RETURN_ON_ERR(cudaMemcpy(panel_state->micro_fringe,A_sym.macro_micro_fringe.data(),A_sym.macro_micro_fringe.size()*sizeof(unsigned),cudaMemcpyHostToDevice),"sparse microtile fringe upload");
        if(panel_state->fringe_offset) CUDA_RETURN_ON_ERR(cudaMemcpy(panel_state->fringe_offset,A_sym.macro_fringe_offset.data(),A_sym.macro_fringe_offset.size()*sizeof(unsigned),cudaMemcpyHostToDevice),"sparse fringe offsets upload");
        if(panel_state->fringe_delta) CUDA_RETURN_ON_ERR(cudaMemcpy(panel_state->fringe_delta,A_sym.macro_fringe_delta.data(),A_sym.macro_fringe_delta.size()*sizeof(unsigned short),cudaMemcpyHostToDevice),"sparse fringe operands upload");
        if(panel_state->micro_delta) CUDA_RETURN_ON_ERR(cudaMemcpy(panel_state->micro_delta,A_sym.macro_micro_delta.data(),A_sym.macro_micro_delta.size()*sizeof(unsigned short),cudaMemcpyHostToDevice),"sparse microtile operands upload");
        if(panel_state->normalize_position) CUDA_RETURN_ON_ERR(cudaMemcpy(panel_state->normalize_position,A_sym.macro_normalize_position.data(),A_sym.macro_normalize_position.size()*sizeof(unsigned),cudaMemcpyHostToDevice),"sparse flat normalization positions upload");
        if(panel_state->normalize_diagonal) CUDA_RETURN_ON_ERR(cudaMemcpy(panel_state->normalize_diagonal,A_sym.macro_normalize_diagonal.data(),A_sym.macro_normalize_diagonal.size()*sizeof(unsigned),cudaMemcpyHostToDevice),"sparse flat normalization diagonal upload");
        CUDA_RETURN_ON_ERR(cudaMemset(panel_state->info,0,sizeof(int)),"sparse panel info initialization");
    }
    if (destination_owned) {
        CUDA_RETURN_ON_ERR(cudaMemcpy(destination_state->lower, A_sym.destination_lower.data(), A_sym.destination_lower.size() * sizeof(unsigned), cudaMemcpyHostToDevice), "destination lower upload");
        if(destination_packed)
            CUDA_RETURN_ON_ERR(cudaMemcpy(destination_state->upper_delta,A_sym.destination_upper_delta.data(),A_sym.destination_upper_delta.size()*sizeof(unsigned short),cudaMemcpyHostToDevice),"destination relative upper upload");
        else CUDA_RETURN_ON_ERR(cudaMemcpy(destination_state->upper, A_sym.destination_upper.data(), A_sym.destination_upper.size() * sizeof(unsigned), cudaMemcpyHostToDevice), "destination upper upload");
        CUDA_RETURN_ON_ERR(cudaMemcpy(destination_state->pointer, A_sym.destination_group_ptr.data(), A_sym.destination_group_ptr.size() * sizeof(unsigned), cudaMemcpyHostToDevice), "destination pointer upload");
        CUDA_RETURN_ON_ERR(cudaMemcpy(destination_state->target, A_sym.destination_target.data(), A_sym.destination_target.size() * sizeof(unsigned), cudaMemcpyHostToDevice), "destination target upload");
        if (destination_flat) {
            CUDA_RETURN_ON_ERR(cudaMemcpy(destination_state->normalize_lower,A_sym.destination_normalize_lower.data(),A_sym.destination_normalize_lower.size()*sizeof(unsigned),cudaMemcpyHostToDevice), "root lower upload");
            CUDA_RETURN_ON_ERR(cudaMemcpy(destination_state->normalize_diagonal,A_sym.destination_normalize_diagonal.data(),A_sym.destination_normalize_diagonal.size()*sizeof(unsigned),cudaMemcpyHostToDevice), "root diagonal upload");
        }
        if(destination_cta) {
            CUDA_RETURN_ON_ERR(cudaMemcpy(destination_state->normalize_level,A_sym.destination_normalize_ptr.data(),A_sym.destination_normalize_ptr.size()*sizeof(unsigned),cudaMemcpyHostToDevice),"gather normalization prefix upload");
            CUDA_RETURN_ON_ERR(cudaMemcpy(destination_state->group_level,A_sym.destination_level_ptr.data(),A_sym.destination_level_ptr.size()*sizeof(unsigned),cudaMemcpyHostToDevice),"gather group prefix upload");
            if(destination_state->heavy_level)
                CUDA_RETURN_ON_ERR(cudaMemcpy(destination_state->heavy_level,A_sym.destination_heavy_level_ptr.data(),A_sym.destination_heavy_level_ptr.size()*sizeof(unsigned),cudaMemcpyHostToDevice),"gather heavy prefix upload");
        }
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
                        l_col_ptr_dev, level_idx_dev, scratch_tag_dev, tmpMem, n, source_head, PERTURB, pert, nullptr, 0, deferred_cut);
                    if (!capturing) after_launch();
                } else {
                    const unsigned cache_first = A_sym.cached_prepare_ptr[source_head];
                    const unsigned cache_end = A_sym.cached_prepare_ptr[source_head + rest_col];
                    const unsigned scatter_first = A_sym.scratch_prepare_ptr[source_head];
                    const unsigned scatter_end = A_sym.scratch_prepare_ptr[source_head + rest_col];
                    if (cache_end > cache_first) {
                        if (!capturing) before_launch();
                        RL_cached_source_prepare<<<(cache_end - cache_first + 3) / 4, 4 * deviceProp.warpSize, 0, launch_stream>>>(
                            sym_c_ptr_dev, val_dev, l_col_ptr_dev, level_idx_dev, cached_prepare_sources_dev, cache_first, cache_end, deferred_cut);
                        if (!capturing) after_launch();
                    }
                    if (scatter_end > scatter_first) {
                        if (!capturing) before_launch();
                        RL_batch_prepare<<<scatter_end - scatter_first, 256, 0, launch_stream>>>(sym_c_ptr_dev, sym_r_idx_dev, val_dev,
                            l_col_ptr_dev, level_idx_dev, scratch_tag_dev, tmpMem, n, source_head, false, pert,
                            scratch_prepare_sources_dev, scatter_first, deferred_cut);
                        if (!capturing) after_launch();
                    }
                }
                if (tasks) {
                    if (!capturing) before_launch();
                    RL_update_tasks<<<(tasks + 3) / 4, 4 * deviceProp.warpSize, 0, launch_stream>>>(sym_c_ptr_dev,
                    sym_r_idx_dev, val_dev, l_col_ptr_dev, csr_c_idx_dev, level_idx_dev,
                    csr_pivot_pos_dev, batch_task_entry_dev, batch_task_source_dev, batch_task_begin_dev, batch_task_end_dev, update_pair_begin_dev, update_pair_end_dev, update_pair_source_dev, update_pair_target_dev, scratch_tag_dev, tmpMem, n, source_head, task_head, task_head + tasks, packed_short, deferred_cut);
                    if (!capturing) after_launch();
                }
                if (packed_short) RL_update_short_tasks<<<(short_tasks + 15) / 16, 256, 0, launch_stream>>>(
                    val_dev, csr_pivot_pos_dev, batch_task_entry_dev, short_update_task_index_dev,
                    update_pair_begin_dev, update_pair_end_dev, update_pair_source_dev, update_pair_target_dev,
                    batch_task_source_dev, short_head, short_head + short_tasks, sym_r_idx_dev, deferred_cut);
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
                                                    in_level_pos, deferred_cut);
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
            if (destination_flat) {
                const unsigned first=A_sym.destination_normalize_ptr[lv];
                const unsigned entries=A_sym.destination_normalize_ptr[lv+1]-first;
                // A zero-count node preserves capture structure without any arithmetic.
                const unsigned blocks=std::max(1u,(entries+255)/256);
                destination_entry_normalize<<<blocks,256,0,stream>>>(
                    val_dev,destination_state->normalize_lower+first,destination_state->normalize_diagonal+first,entries);
            }
            else destination_source_normalize<<<(count + 3) / 4, 4 * deviceProp.warpSize, 0, stream>>>(
                val_dev, sym_c_ptr_dev, l_col_ptr_dev, level_idx_dev, source_first, source_end);
            if (!capturing) after_launch();
        }
        const unsigned first = A_sym.destination_level_ptr[lv], end = A_sym.destination_level_ptr[lv + 1];
        const unsigned split=A_sym.destination_heavy_level_ptr.empty()?end:A_sym.destination_heavy_level_ptr[lv];
        if (split > first) {
            if(destination_packed)
                destination_group_reduce_relative<<<(split-first+255)/256,256,0,stream>>>(
                    val_dev,destination_state->pointer,destination_state->target,
                    destination_state->lower,destination_state->upper_delta,first,split);
            else destination_group_reduce<<<(split - first + 255) / 256, 256, 0, stream>>>(
                val_dev, destination_state->pointer, destination_state->target,
                destination_state->lower, destination_state->upper, first, split);
        }
        if (end>split) {
            const unsigned blocks=(end-split+3)/4;
            if (destination_packed)
                destination_group_reduce_warp<true><<<blocks,4*deviceProp.warpSize,0,stream>>>(
                    val_dev,destination_state->pointer,destination_state->target,destination_state->lower,
                    destination_state->upper,destination_state->upper_delta,split,end);
            else destination_group_reduce_warp<false><<<blocks,4*deviceProp.warpSize,0,stream>>>(
                    val_dev,destination_state->pointer,destination_state->target,destination_state->lower,
                    destination_state->upper,destination_state->upper_delta,split,end);
        }
    };

    auto launch_panel_singletons = [&](unsigned lv,cudaStream_t stream) {
        const unsigned xa=A_sym.macro_singleton_ptr[lv],xb=A_sym.macro_singleton_ptr[lv+1];
        if(A_sym.macro_singleton_compiled) {
            const unsigned ma=A_sym.macro_singleton_mapped_ptr[lv],mb=A_sym.macro_singleton_mapped_ptr[lv+1];
            const unsigned split=A_sym.macro_work_ready?A_sym.macro_mapped_short_end[lv]:ma;
            if(split>ma) sparse_singletons_compiled_short<<<(split-ma+15)/16,256,0,stream>>>(val_dev,
                sym_c_ptr_dev,sym_r_idx_dev,l_col_ptr_dev,csr_pivot_pos_dev,panel_state->singleton_entry,
                panel_state->singleton_mapped,panel_state->singleton_offset,panel_state->singleton_delta,ma,split);
            if(mb>split) sparse_singletons_compiled<<<(mb-split+3)/4,4*deviceProp.warpSize,0,stream>>>(val_dev,
                sym_c_ptr_dev,sym_r_idx_dev,l_col_ptr_dev,csr_pivot_pos_dev,panel_state->singleton_entry,
                panel_state->singleton_mapped,panel_state->singleton_offset,panel_state->singleton_delta,split,mb);
            const unsigned fa=A_sym.macro_singleton_fallback_ptr[lv],fb=A_sym.macro_singleton_fallback_ptr[lv+1];
            if(fb>fa) sparse_singletons_queued_fallback<<<(fb-fa+3)/4,4*deviceProp.warpSize,0,stream>>>(val_dev,
                sym_c_ptr_dev,sym_r_idx_dev,l_col_ptr_dev,csr_c_idx_dev,csr_pivot_pos_dev,
                panel_state->singleton_entry,panel_state->singleton_fallback,fa,fb,panel_state->info);
        } else if(xb>xa) sparse_panel_singletons<<<(xb-xa+3)/4,4*deviceProp.warpSize,0,stream>>>(val_dev,
                sym_c_ptr_dev,sym_r_idx_dev,l_col_ptr_dev,csr_c_idx_dev,csr_pivot_pos_dev,
                panel_state->singleton_entry,xa,xb,panel_state->info);
    };
    auto launch_panel_level = [&](unsigned lv,cudaStream_t stream) {
        if(A_sym.macro_work_ready) {
            const unsigned la=A_sym.macro_multi_lower_begin[lv],lb=A_sym.macro_lower_ptr[lv+1];
            const unsigned ua=A_sym.macro_upper_ptr[lv],ub=A_sym.macro_upper_ptr[lv+1];
            const unsigned prepare_count=(lb-la)+(ub-ua);
            if(prepare_count) sparse_panel_prepare_local<<<prepare_count,256,0,stream>>>(val_dev,
                sym_c_ptr_dev,l_col_ptr_dev,csr_r_ptr_dev,csr_diag_ptr_dev,csr_pivot_pos_dev,
                panel_state->head,panel_state->width,panel_state->lower_node,panel_state->lower_first,
                panel_state->upper_node,panel_state->upper_first,la,lb-la,ua,panel_state->info);
            const unsigned da=A_sym.macro_diagonal_ptr[lv],db=A_sym.macro_diagonal_ptr[lv+1];
            const unsigned fa=A_sym.macro_microtile_ready?A_sym.macro_micro_fringe_ptr[lv]:A_sym.macro_schur_ptr[lv];
            const unsigned fb=A_sym.macro_microtile_ready?A_sym.macro_micro_fringe_ptr[lv+1]:A_sym.macro_schur_ptr[lv+1];
            const unsigned na=A_sym.macro_normalize_ptr[lv],nb=A_sym.macro_normalize_ptr[lv+1];
            const unsigned commit_count=(db-da)+(fb-fa)+(nb-na+255)/256;
            const unsigned *fringe=A_sym.macro_microtile_ready?panel_state->micro_fringe:nullptr;
            if(commit_count) sparse_panel_commit_fringe_norm<<<commit_count,256,0,stream>>>(val_dev,
                sym_c_ptr_dev,sym_r_idx_dev,l_col_ptr_dev,csr_r_ptr_dev,csr_c_idx_dev,csr_diag_ptr_dev,
                csr_pivot_pos_dev,panel_state->head,panel_state->width,panel_state->diagonal,da,db-da,
                panel_state->schur_node,panel_state->schur_row,panel_state->schur_col,fringe,fa,fb-fa,
                panel_state->normalize_position,panel_state->normalize_diagonal,na,nb,
                panel_state->fringe_offset,panel_state->fringe_delta,panel_state->info);
            if(A_sym.macro_microtile_ready) {
                const unsigned ma=A_sym.macro_micro_ptr[lv],mb=A_sym.macro_micro_ptr[lv+1];
                if(mb>ma) sparse_panel_microtiles<<<mb-ma,256,0,stream>>>(val_dev,l_col_ptr_dev,csr_diag_ptr_dev,
                    csr_pivot_pos_dev,panel_state->head,panel_state->width,panel_state->micro_node,
                    panel_state->micro_row,panel_state->micro_col,panel_state->micro_delta,ma);
            }
            launch_panel_singletons(lv,stream);
            return;
        }
        const unsigned da=A_sym.macro_diagonal_ptr[lv],db=A_sym.macro_diagonal_ptr[lv+1];
        if(db>da) sparse_panel_diagonal<<<db-da,256,0,stream>>>(val_dev,l_col_ptr_dev,
            panel_state->head,panel_state->width,panel_state->diagonal,da,panel_state->info);
        const unsigned la=A_sym.macro_lower_ptr[lv],lb=A_sym.macro_lower_ptr[lv+1];
        if(lb>la) sparse_panel_lower<<<lb-la,256,0,stream>>>(val_dev,sym_c_ptr_dev,l_col_ptr_dev,
            panel_state->head,panel_state->width,panel_state->lower_node,panel_state->lower_first,la);
        const unsigned ua=A_sym.macro_upper_ptr[lv],ub=A_sym.macro_upper_ptr[lv+1];
        if(ub>ua) sparse_panel_upper<<<ub-ua,256,0,stream>>>(val_dev,l_col_ptr_dev,csr_r_ptr_dev,
            csr_diag_ptr_dev,csr_pivot_pos_dev,panel_state->head,panel_state->width,
            panel_state->upper_node,panel_state->upper_first,ua);
        const unsigned sa=A_sym.macro_schur_ptr[lv],sb=A_sym.macro_schur_ptr[lv+1];
        if(A_sym.macro_microtile_ready) {
            const unsigned ma=A_sym.macro_micro_ptr[lv],mb=A_sym.macro_micro_ptr[lv+1];
            if(mb>ma) sparse_panel_microtiles<<<mb-ma,256,0,stream>>>(val_dev,l_col_ptr_dev,csr_diag_ptr_dev,
                csr_pivot_pos_dev,panel_state->head,panel_state->width,panel_state->micro_node,
                panel_state->micro_row,panel_state->micro_col,panel_state->micro_delta,ma);
            const unsigned fa=A_sym.macro_micro_fringe_ptr[lv],fb=A_sym.macro_micro_fringe_ptr[lv+1];
            if(fb>fa) sparse_panel_queued_fringe<<<fb-fa,dim3(16,16),0,stream>>>(val_dev,sym_c_ptr_dev,sym_r_idx_dev,
                l_col_ptr_dev,csr_r_ptr_dev,csr_c_idx_dev,csr_diag_ptr_dev,csr_pivot_pos_dev,
                panel_state->head,panel_state->width,panel_state->schur_node,panel_state->schur_row,
                panel_state->schur_col,panel_state->micro_fringe,fa,panel_state->info);
        } else if(sb>sa) sparse_panel_schur<<<sb-sa,dim3(16,16),0,stream>>>(val_dev,sym_c_ptr_dev,sym_r_idx_dev,
            l_col_ptr_dev,csr_r_ptr_dev,csr_c_idx_dev,csr_diag_ptr_dev,csr_pivot_pos_dev,
            panel_state->head,panel_state->width,panel_state->schur_node,panel_state->schur_row,
            panel_state->schur_col,sa,panel_state->info);
        launch_panel_singletons(lv,stream);
    };
    const auto panel_level=[&](unsigned lv) {return panel_state && lv>=1 && lv<A_sym.macro_end_level;};

    // Build only a reusable schedule: stream capture enqueues no numerical
    // execution and reads no matrix values. Numeric work starts below.
    const double graph_prepare_start = stamp();
    std::vector<unsigned> persistent_end(num_lev, 0);
    std::vector<unsigned char> narrow_ok(num_lev, 0), persistent_covered(num_lev, 0);
    for (unsigned lv = 1; lv < num_lev; ++lv) {
        if(panel_level(lv)) continue;
        if (dense_level(lv) || (deferred_active && A_sym.dense_deferred_level_count[lv])) continue;
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
        graph_eligible[lv] = !panel_level(lv) && !dense_level(lv) && !persistent_covered[lv] &&
            // Generic kernels already skip selected sources. Capture mixed
            // levels; completely selected levels remain host-skipped. The
            // separate unmasked narrow-chain eligibility guard stays intact.
            (!deferred_active || A_sym.dense_deferred_level_count[lv]<count) && (count > Nstreams || small ||
            A_sym.batch_task_ptr[head + count] > A_sym.batch_task_ptr[head]);
    }
    if (destination_whole) {
        unsigned cta_runs=0,cta_levels=0;
        const auto fits_one_cta=[&](unsigned lv) {
            if(!destination_cta) return false;
            const unsigned na=A_sym.destination_normalize_ptr[lv],nb=A_sym.destination_normalize_ptr[lv+1];
            const unsigned first=A_sym.destination_level_ptr[lv],end=A_sym.destination_level_ptr[lv+1];
            const unsigned split=A_sym.destination_heavy_level_ptr.empty()?end:A_sym.destination_heavy_level_ptr[lv];
            return nb-na<=256 && split-first<=256 && end-split<=256/deviceProp.warpSize;
        };
        CUDA_RETURN_ON_ERR(cudaStreamCreateWithFlags(&capture_stream, cudaStreamNonBlocking), "destination capture stream");
        for (unsigned head = 1; head < num_lev;) {
            const unsigned end = std::min(head + 128u, num_lev);
            CUDA_RETURN_ON_ERR(cudaStreamBeginCapture(capture_stream, cudaStreamCaptureModeGlobal), "destination capture begin");
            capture_active = true;
            for (unsigned lv = head; lv < end;) {
                if (persistent_end[lv]) {
                    const unsigned chain_end = std::min(persistent_end[lv], end);
                    // Reuse the already proven bounded single-CTA program. All
                    // levels and external updates stay inside numeric execution.
                    RL_narrow_chain<<<1, 1024, 0, capture_stream>>>(
                        sym_c_ptr_dev, sym_r_idx_dev, val_dev, l_col_ptr_dev,
                        csr_r_ptr_dev, csr_c_idx_dev, csr_diag_ptr_dev, csr_pivot_pos_dev,
                        level_idx_dev, scratch_tag_dev, tmpMem, n, level_ptr_dev, lv, chain_end);
                    lv = chain_end;
                } else if(fits_one_cta(lv)) {
                    unsigned run_end=lv+1;
                    while(run_end<end && !persistent_end[run_end] && fits_one_cta(run_end)) ++run_end;
                    if(destination_packed)
                        destination_gather_level_run<true><<<1,256,0,capture_stream>>>(val_dev,
                            destination_state->normalize_level,destination_state->group_level,destination_state->heavy_level,
                            destination_state->normalize_lower,destination_state->normalize_diagonal,
                            destination_state->pointer,destination_state->target,destination_state->lower,
                            destination_state->upper,destination_state->upper_delta,lv,run_end);
                    else destination_gather_level_run<false><<<1,256,0,capture_stream>>>(val_dev,
                            destination_state->normalize_level,destination_state->group_level,destination_state->heavy_level,
                            destination_state->normalize_lower,destination_state->normalize_diagonal,
                            destination_state->pointer,destination_state->target,destination_state->lower,
                            destination_state->upper,destination_state->upper_delta,lv,run_end);
                    ++cta_runs;cta_levels+=run_end-lv;lv=run_end;
                } else {
                    launch_destination_level(lv, capture_stream, true);
                    ++lv;
                }
            }
            const cudaError_t status = cudaStreamEndCapture(capture_stream, &level_graph[head]);
            capture_active = false;
            CUDA_RETURN_ON_ERR(status, "destination capture end");
            CUDA_RETURN_ON_ERR(cudaGraphInstantiate(&level_exec[head], level_graph[head], nullptr, nullptr, 0), "destination graph instantiate");
            graph_end[head] = end; ++graph_segments; graph_levels += end - head; head = end;
        }
        out << "CTA destination runs: " << cta_runs << ", levels: " << cta_levels
            << ", prefix bytes: " << (destination_cta?4ull*(A_sym.destination_normalize_ptr.size()+
                A_sym.destination_level_ptr.size()+A_sym.destination_heavy_level_ptr.size()):0ull) << endl;
    }
    if (!PERTURB && !destination_whole) {
        CUDA_RETURN_ON_ERR(cudaStreamCreateWithFlags(&capture_stream, cudaStreamNonBlocking), "graph capture stream");
        if (dense_size) {
            CUDA_RETURN_ON_ERR(cudaStreamBeginCapture(capture_stream, cudaStreamCaptureModeGlobal), "dense panel capture begin");
            capture_active = true;
            unsigned kernels = 0;
            for (unsigned first = 0; first < dense_size; first += 16) {
                const unsigned end = std::min(first + 16u, dense_size);
                dense_diagonal_shared<<<1, 256, 0, capture_stream>>>(dense_state->values, dense_size, first, end, dense_state->info);
                ++kernels;
                if (end < dense_size) {
                    const unsigned tail = dense_size - end;
                    dense_panel_lower_register<<<(tail+255)/256,256,0,capture_stream>>>(
                        dense_state->values,dense_size,first,end,dense_state->info);
                    ++kernels;
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
            out << "Shared-resident diagonal / reused register lower solve enabled" << endl;
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
    if(panel_state) {
        for(unsigned head=0;head<A_sym.macro_num_levels;head+=128) {
            const unsigned end=std::min(head+128u,A_sym.macro_num_levels);
            panel_state->graphs.push_back(nullptr);panel_state->execs.push_back(nullptr);
            CUDA_RETURN_ON_ERR(cudaStreamBeginCapture(capture_stream,cudaStreamCaptureModeGlobal),"sparse panel capture begin");
            capture_active=true;
            for(unsigned lv=head;lv<end;++lv) launch_panel_level(lv,capture_stream);
            const cudaError_t status=cudaStreamEndCapture(capture_stream,&panel_state->graphs.back());
            capture_active=false;
            CUDA_RETURN_ON_ERR(status,"sparse panel capture end");
            CUDA_RETURN_ON_ERR(cudaGraphInstantiate(&panel_state->execs.back(),panel_state->graphs.back(),nullptr,nullptr,0),"sparse panel graph instantiate");
        }
        out << "Sparse panel execution: macro levels " << A_sym.macro_num_levels
            << ", graphs " << panel_state->execs.size() << ", original region [1," << A_sym.macro_end_level << ")" << endl;
    }
    CUDA_RETURN_ON_ERR(cudaDeviceSynchronize(), "graph preparation synchronize");
    record("structure_preparation", graph_prepare_start, true);
    const double factor_start = stamp();

    unsigned expanded_batch_levels = 0;
    bool pending_batched = false;
    if (destination_whole) {
        launch_destination_level(0, 0, false);
        if (!first_kernel_ok) { cleanup(); return; }
        CUDA_RETURN_ON_ERR(cudaGetLastError(), "destination level zero launch");
        for (unsigned head = 1; head < num_lev; head = graph_end[head])
            CUDA_RETURN_ON_ERR(cudaGraphLaunch(level_exec[head], 0), "destination graph launch");
        pending_batched = true;
    }
    for (unsigned i = 0; !destination_whole && i < num_lev; ++i)
    {
        if(panel_state && i==1) {
            CUDA_RETURN_ON_ERR(cudaDeviceSynchronize(),"sparse panel root predecessors");
            before_launch();
            for(auto exec:panel_state->execs) CUDA_RETURN_ON_ERR(cudaGraphLaunch(exec,0),"sparse panel graph launch");
            after_launch();
            if(!first_kernel_ok) {cleanup();return;}
            int info=0;
            CUDA_RETURN_ON_ERR(cudaMemcpy(&info,panel_state->info,sizeof(int),cudaMemcpyDeviceToHost),"sparse panel numeric status");
            if(info) {err << "Sparse panel invalid pivot or absent target at column " << info-1 << endl;cleanup();return;}
            pending_batched=true;i=A_sym.macro_end_level-1;
            continue;
        }
        int lev_size = A_sym.level_ptr[i + 1] - A_sym.level_ptr[i];
        if (lev_size <= 0)
            continue;
        if(deferred_active && A_sym.dense_deferred_level_count[i]==unsigned(lev_size)) {
            ++deferred_skipped_levels;
            continue;
        }
        if (destination_root && i == 0) {
            launch_destination_level(0,0,false);
            if (!first_kernel_ok) {cleanup();return;}
            CUDA_RETURN_ON_ERR(cudaGetLastError(),"root destination level launch");
            pending_batched=true;
            continue;
        }
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
                        if(currentCol>=deferred_cut) continue;
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

    if(deferred_active) {
        // Every external/prefix source is complete before reading the Schur block.
        CUDA_RETURN_ON_ERR(cudaDeviceSynchronize(), "deferred dense prefix completion");
        const unsigned entries=dense_size*dense_size;
        CUDA_RETURN_ON_ERR(cudaMemset(dense_state->info,0,sizeof(int)), "deferred dense info reset");
        before_launch();
        dense_tail_pack<<<(entries+255)/256,256>>>(val_dev,dense_state->positions,dense_state->values,entries);
        after_launch();
        if(!first_kernel_ok) {cleanup();return;}
        CUDA_RETURN_ON_ERR(cudaGraphLaunch(dense_state->exec,0), "deferred dense panel graph launch");
        dense_tail_unpack<<<(entries+255)/256,256>>>(val_dev,dense_state->positions,dense_state->values,entries);
        CUDA_RETURN_ON_ERR(cudaGetLastError(), "deferred dense pack/unpack launches");
        int info=-1;
        CUDA_RETURN_ON_ERR(cudaMemcpy(&info,dense_state->info,sizeof(int),cudaMemcpyDeviceToHost), "deferred dense info download");
        if(info) {err << "deferred dense zero pivot: " << info << endl;cleanup();return;}
        pending_batched=true;
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
    if(deferred_active) out << "Deferred dense skipped source levels: " << deferred_skipped_levels << endl;
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
        panel_state.reset();
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
