#ifndef _SYMBOLIC_H_
#define _SYMBOLIC_H_
#include <vector>
#include <memory>
#include <unordered_set>
#include <iostream>
#include "type.h"
#include "nicslu.h"

class Symbolic_Matrix
{
public:
    unsigned int n;
    unsigned int nnz;
    int num_lev;
    std::vector<unsigned> sym_c_ptr;
    std::vector<unsigned> sym_r_idx;
    std::vector<unsigned> csr_r_ptr;
    std::vector<unsigned> csr_c_idx;
    std::vector<unsigned> csr_diag_ptr;
    std::vector<unsigned> csr_pivot_pos; // structure-only CSR-to-CSC position map
    std::vector<REAL> val;
    std::vector<unsigned> numeric_scatter_src, numeric_scatter_dst;
    bool numeric_scatter_ready = false;
    bool numeric_device_scatter = false;
    std::vector<REAL> numeric_packed;
    void prepare_numeric_scatter(unsigned*, unsigned*);
    std::vector<unsigned> l_col_ptr; //Indices of diagonal elements
    std::vector<int> level_idx;
    std::vector<int> level_ptr;
    std::vector<unsigned> dense_columns, dense_positions;
    int dense_first_level = -1, dense_end_level = -1;
    void prepare_dense_tail(const std::vector<int>&);
    void prepare_near_dense_tail(const std::vector<int>&);
    std::vector<unsigned> batch_task_ptr, batch_task_entry, batch_task_source;
    std::vector<unsigned> batch_task_begin, batch_task_end;
    std::vector<unsigned> update_pair_begin, update_pair_end, update_pair_source, update_pair_target;

    void predictLU(unsigned*, unsigned*, double*);
    void csr();
    void leveling();
    void fill_in(unsigned*, unsigned*);
    std::vector<REAL> solve(SNicsLU*, const std::vector<REAL> &);
    std::vector<REAL> solve_CSR(SNicsLU*, const std::vector<REAL> &);

#if GLU_DEBUG
    void PrintLevel();
    //ABFT
    std::vector<REAL> CCA;
    void ABFTCalculateCCA();
    void ABFTCheckResult();
#endif

    Symbolic_Matrix(unsigned n, std::ostream &out, std::ostream &err):
        n(n),
        num_lev(0),
        m_out(out),
        m_err(err) {};

    Symbolic_Matrix(std::ostream &out, std::ostream &err):
        m_out(out), m_err(err) {};

private:
    std::ostream &m_out;
    std::ostream &m_err;
public:
    // Declared last so resources are released before factor storage on destruction.
    std::shared_ptr<void> gpu_resources;
};


#endif
