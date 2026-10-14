#include <iostream>
#include <vector>
#include <algorithm>
#include <iterator>
#include <functional>
#include "Timer.h"
#include "symbolic.h"
#include "near_cached_core.h"
#include <cmath>

using namespace std;

void Symbolic_Matrix :: fill_in(unsigned *ai, unsigned *ap)
{
    numeric_scatter_ready = false;
    numeric_device_scatter = false;
    sym_c_ptr.push_back(0);
    vector<unsigned> mark(n, n), pending, tmpcol1, tmpcol2;
    bool preserve_multiplicity = false;
    for (unsigned i = 0; i < n; ++i) {
        tmpcol2.assign(ai + ap[i], ai + ap[i + 1]);
        sort(tmpcol2.begin(), tmpcol2.end());
        // The legacy set-union path preserves input multiplicity. If any
        // duplicate enters the structure, retain that path for later columns.
        preserve_multiplicity = preserve_multiplicity ||
            adjacent_find(tmpcol2.begin(), tmpcol2.end()) != tmpcol2.end();
        if (preserve_multiplicity) {
            for (unsigned j = 0; j < tmpcol2.size(); ++j) {
                const unsigned nz_idx = tmpcol2[j];
                if (nz_idx >= i) break;
                auto colhead = sym_r_idx.begin() + l_col_ptr[nz_idx];
                auto colend = sym_r_idx.begin() + sym_c_ptr[nz_idx + 1];
                tmpcol1.resize(distance(colhead, colend) + tmpcol2.size());
                auto it = set_union(colhead, colend, tmpcol2.begin(), tmpcol2.end(), tmpcol1.begin());
                tmpcol1.resize(it - tmpcol1.begin());
                tmpcol2.clear();
                swap(tmpcol1, tmpcol2);
            }
        } else {
            pending.clear();
            for (unsigned row : tmpcol2) {
                mark[row] = i;
                if (row < i) pending.push_back(row);
            }
            make_heap(pending.begin(), pending.end(), greater<unsigned>());
            // Prior symbolic lower columns contain only rows >= their pivot.
            // Process newly discovered upper dependencies in increasing order.
            while (!pending.empty()) {
                const unsigned col = pending.front();
                pop_heap(pending.begin(), pending.end(), greater<unsigned>());
                pending.pop_back();
                for (unsigned k = l_col_ptr[col]; k < sym_c_ptr[col + 1]; ++k) {
                    const unsigned row = sym_r_idx[k];
                    if (mark[row] == i) continue;
                    mark[row] = i;
                    tmpcol2.push_back(row);
                    if (row < i) {
                        pending.push_back(row);
                        push_heap(pending.begin(), pending.end(), greater<unsigned>());
                    }
                }
            }
            sort(tmpcol2.begin(), tmpcol2.end());
        }
        auto diag_it = lower_bound(tmpcol2.begin(), tmpcol2.end(), i);
        if (diag_it == tmpcol2.end() || *diag_it != i)
            diag_it = tmpcol2.insert(diag_it, i);
        l_col_ptr.push_back(sym_c_ptr.back() + distance(tmpcol2.begin(), diag_it));
        sym_c_ptr.push_back(tmpcol2.size() + sym_c_ptr.back());
        sym_r_idx.insert(sym_r_idx.end(), tmpcol2.begin(), tmpcol2.end());
    }
    nnz = sym_c_ptr.back();
    m_out << "Symbolic nonzero: " << nnz << endl;
}

void Symbolic_Matrix :: csr()
{
    csr_r_ptr.push_back(0);
    vector<vector<unsigned> > buffer(n, vector<unsigned>());

    for (unsigned i = 0; i < n; ++i)
    {
        for (unsigned j = sym_c_ptr[i]; j < sym_c_ptr[i+1]; ++j)
        {
            unsigned row_idx = sym_r_idx[j];
            buffer[row_idx].push_back(i);
        }
    }

    for (unsigned i = 0; i < n; ++i)
    {
        for (unsigned j = 0; j < buffer[i].size(); ++j)
        {
            csr_c_idx.push_back(buffer[i][j]);
            if(buffer[i][j] == i)
                csr_diag_ptr.push_back(csr_c_idx.size()-1);
        }
        csr_r_ptr.push_back(csr_r_ptr.back() + buffer[i].size());
    }
    // CSR entries were emitted in CSC-column order. Replay that order to
    // bind each entry to its exact CSC position, using only symbolic indices.
    csr_pivot_pos.resize(csr_c_idx.size());
    vector<unsigned> next(csr_r_ptr.begin(), csr_r_ptr.end() - 1);
    for (unsigned col = 0; col < n; ++col) {
        bool duplicates = false;
        for (unsigned k = sym_c_ptr[col] + 1; k < sym_c_ptr[col + 1]; ++k)
            if (sym_r_idx[k - 1] == sym_r_idx[k]) { duplicates = true; break; }
        for (unsigned k = sym_c_ptr[col]; k < sym_c_ptr[col + 1]; ++k)
            csr_pivot_pos[next[sym_r_idx[k]]++] = duplicates ? ~0u : k;
    }
}

// Reusable index-only plan; duplicate inputs keep the first matching entry.
void Symbolic_Matrix::prepare_numeric_scatter(unsigned *ai, unsigned *ap) {
    numeric_scatter_src.clear(); numeric_scatter_dst.clear();
    vector<unsigned> position(n, 0), row_stamp(n, n);
    for (unsigned i = 0; i < n; ++i) {
        for (unsigned k = ap[i]; k < ap[i + 1]; ++k) {
            const unsigned col = ai[k];
            if (row_stamp[col] != i) { row_stamp[col] = i; position[col] = k; }
        }
        for (unsigned j = sym_c_ptr[i]; j < sym_c_ptr[i + 1]; ++j) {
            const unsigned col = sym_r_idx[j];
            if (row_stamp[col] == i) {
                numeric_scatter_dst.push_back(j);
                numeric_scatter_src.push_back(position[col]);
            }
        }
    }
    numeric_scatter_ready = true;
}

//Construct val vector (including filled in zeros) of symbolic_matrix
void Symbolic_Matrix :: predictLU (unsigned *ai, unsigned *ap, double *ax)
{
    if (numeric_device_scatter && numeric_scatter_ready) {
        numeric_packed.resize(numeric_scatter_src.size());
        for (size_t j = 0; j < numeric_scatter_src.size(); ++j)
            numeric_packed[j] = REAL(ax[numeric_scatter_src[j]]);
        return;
    }
    if (numeric_scatter_ready) {
        val.assign(nnz, REAL(0));
        for (size_t j = 0; j < numeric_scatter_src.size(); ++j)
            val[numeric_scatter_dst[j]] = REAL(ax[numeric_scatter_src[j]]);
        return;
    }
    // Stamp each original column index once per row. Keep the first match,
    // exactly as std::find did, even when an input row contains duplicates.
    vector<unsigned> position(n, 0), row_stamp(n, n);
    val.resize(nnz);
    for (unsigned i = 0; i < n; ++i) {
        for (unsigned k = ap[i]; k < ap[i + 1]; ++k) {
            const unsigned col = ai[k];
            if (row_stamp[col] != i) {
                row_stamp[col] = i;
                position[col] = k;
            }
        }
        for (unsigned j = sym_c_ptr[i]; j < sym_c_ptr[i + 1]; ++j) {
            const unsigned col = sym_r_idx[j];
            val[j] = row_stamp[col] == i ? (REAL)ax[position[col]] : (REAL)0;
        }
    }
}


void Symbolic_Matrix::prepare_dense_tail(const std::vector<int>& levels)
{
    dense_columns.clear(); dense_positions.clear();
    dense_first_level = dense_end_level = -1;
    // A complete k-node tail contains a k-node U-dependency chain.
    // These bounds reject impossible inputs without detector allocations.
    if (n < 512 || num_lev < 512) {
        m_out << "Closed dense LU tail: 0, screen: dimension/depth" << endl;
        return;
    }
    std::vector<unsigned> lower(n);
    unsigned maximum_lower = 0;
    for (unsigned c = 0; c < n; ++c) {
        lower[c] = sym_c_ptr[c + 1] - l_col_ptr[c] - 1;
        maximum_lower = std::max(maximum_lower, lower[c]);
    }
    if (maximum_lower < 511) {
        m_out << "Closed dense LU tail: 0, screen: lower extent" << endl;
        return;
    }
    std::vector<unsigned> upper(n), member(n, ~0u);
    for (unsigned c = 0; c < n; ++c) {
        upper[c] = csr_r_ptr[c + 1] - csr_diag_ptr[c] - 1;
    }
    for (int head = 1; head < num_lev; ++head) {
        unsigned root = ~0u, active = 0;
        for (int p = level_ptr[head]; p < level_ptr[head + 1]; ++p)
            if (lower[level_idx[p]]) { root = level_idx[p]; ++active; }
        if (active != 1) continue;
        const unsigned size = lower[root] + 1;
        if (size < 512 || size > 4096 || size <= dense_columns.size() || upper[root] + 1 != size) continue;
        std::vector<unsigned> cols(1, root);
        cols.insert(cols.end(), sym_r_idx.begin() + l_col_ptr[root] + 1,
                    sym_r_idx.begin() + sym_c_ptr[root + 1]);
        bool ok = std::is_sorted(cols.begin(), cols.end()) &&
                  std::adjacent_find(cols.begin(), cols.end()) == cols.end();
        for (unsigned c : cols) member[c] = head;
        int end = head;
        for (unsigned i = 0; ok && i < size; ++i) {
            const unsigned c = cols[i];
            if (lower[c] != size - i - 1 || upper[c] != size - i - 1) { ok = false; break; }
            for (unsigned j = i + 1; j < size; ++j)
                if (sym_r_idx[l_col_ptr[c] + j - i] != cols[j] ||
                    csr_c_idx[csr_diag_ptr[c] + j - i] != cols[j]) { ok = false; break; }
            if (lower[c]) end = std::max(end, levels[c]);
            // Preserve all conservative dependencies of the bound parent.
            for (unsigned p = sym_c_ptr[c]; ok && p < l_col_ptr[c]; ++p) {
                const unsigned dep = sym_r_idx[p];
                if (lower[dep] && member[dep] != unsigned(head) && levels[dep] >= head) ok = false;
            }
            for (unsigned p = csr_r_ptr[c]; ok && p < csr_diag_ptr[c]; ++p) {
                const unsigned dep = csr_c_idx[p];
                if (member[dep] != unsigned(head) && levels[dep] >= head) ok = false;
            }
        }
        for (int lv = head; ok && lv <= end; ++lv)
            for (int p = level_ptr[lv]; p < level_ptr[lv + 1]; ++p)
                if (lower[level_idx[p]] && member[level_idx[p]] != unsigned(head)) { ok = false; break; }
        if (!ok) continue;
        std::vector<unsigned> positions;
        positions.reserve(size_t(size) * size);
        for (unsigned c : cols) for (unsigned row : cols) {
            auto begin = sym_r_idx.begin() + sym_c_ptr[c];
            auto end_it = sym_r_idx.begin() + sym_c_ptr[c + 1];
            auto pos = std::lower_bound(begin, end_it, row);
            if (pos == end_it || *pos != row || (pos + 1 != end_it && pos[1] == row)) { ok = false; break; }
            positions.push_back(pos - sym_r_idx.begin());
        }
        if (!ok || positions.size() != size_t(size) * size) continue;
        dense_columns.swap(cols); dense_positions.swap(positions);
        dense_first_level = head; dense_end_level = end + 1;
        if (size == 4096) break;
    }
    m_out << "Closed dense LU tail: " << dense_columns.size() << ", levels: "
          << dense_first_level << ".." << dense_end_level << endl;
}

void Symbolic_Matrix::prepare_near_dense_tail(const std::vector<int>& inlevel)
{
    const unsigned density_percent = 95;
    // Necessary global storage bound; valid even without a complete U chain.
    if (n < 512 || size_t(nnz) * 100 < size_t(density_percent) * 512 * 512) {
        m_out << "Near-dense screen: dimension/filled slots" << endl;
        return;
    }
    bool possible_root = false;
    for (unsigned c = 0; c < n; ++c) {
        const unsigned l = sym_c_ptr[c+1] - l_col_ptr[c] - 1;
        const unsigned u = csr_r_ptr[c+1] - csr_diag_ptr[c] - 1;
        if (l && size_t(l) + u + 1 >= 256) { possible_root = true; break; }
    }
    if (!possible_root) {
        m_out << "Near-dense screen: candidate fanout" << endl;
        return;
    }
    std::vector<unsigned> lower(n), upper(n), levels(inlevel.begin(), inlevel.end());
    for (unsigned c = 0; c < n; ++c) {
        lower[c] = sym_c_ptr[c+1] - l_col_ptr[c] - 1;
        upper[c] = csr_r_ptr[c+1] - csr_diag_ptr[c] - 1;
    }
    NearCacheStats stats;
    size_t selected_entries = dense_positions.size();
    scan_near_suffixes(*this, levels, lower, upper, stats,
        [&](const std::vector<unsigned>& component, unsigned cut, size_t entries, int head, int end) {
            const unsigned size = component.size() - cut;
            if (size <= dense_columns.size() || entries * 100 < size_t(size) * size * density_percent) return;
            std::vector<unsigned> cols(component.begin()+cut, component.end()), positions;
            positions.reserve(size_t(size)*size);
            size_t present = 0;
            bool valid = true;
            for (unsigned col : cols) {
                const unsigned finish = sym_c_ptr[col+1];
                unsigned p = std::lower_bound(sym_r_idx.begin()+sym_c_ptr[col],
                                               sym_r_idx.begin()+finish, cols.front()) - sym_r_idx.begin();
                for (unsigned row : cols) {
                    while (p < finish && sym_r_idx[p] < row) ++p;
                    if (p < finish && sym_r_idx[p] == row) {
                        if (p+1 < finish && sym_r_idx[p+1] == row) { valid = false; break; }
                        positions.push_back(p++);
                        ++present;
                    } else {
                        if (row == col) { valid = false; break; }
                        positions.push_back(~0u);
                    }
                }
                if (!valid) break;
            }
            if (!valid || positions.size() != size_t(size)*size || present != entries) return;
            dense_columns.swap(cols);
            dense_positions.swap(positions);
            dense_first_level = head;
            dense_end_level = end;
            selected_entries = present;
        });
    const double density = dense_columns.empty() ? 0.0 :
        double(selected_entries) / (size_t(dense_columns.size()) * dense_columns.size());
    m_out << "Closed near-dense LU tail: " << dense_columns.size()
          << ", density: " << density << ", levels: " << dense_first_level << ".." << dense_end_level
          << ", components: " << stats.components << ", edge visits: " << stats.edge_visits
          << ", budget exhausted: " << stats.budget_exhausted << endl;
}

// Complete symbolic outer-product groups, in the unchanged right-looking DAG.
void Symbolic_Matrix::prepare_destination_groups() {
    destination_ready = false; destination_products = destination_groups = destination_max_fanin = 0;
    destination_level_ptr.clear(); destination_group_ptr.clear(); destination_target.clear();
    destination_lower.clear(); destination_upper.clear();
    if (!n || csr_pivot_pos.size() != nnz) return;
    for (unsigned col = 0; col < n; ++col) {
        if (sym_c_ptr[col + 1] - sym_c_ptr[col] > 1024) return;
        for (unsigned p = sym_c_ptr[col] + 1; p < sym_c_ptr[col + 1]; ++p)
            if (sym_r_idx[p] == sym_r_idx[p - 1]) return;
    }
    const size_t limit = 32u * 1024u * 1024u;
    size_t upper_bound = 0;
    for (unsigned col = 0; col < n; ++col) {
        const unsigned lower = sym_c_ptr[col + 1] - l_col_ptr[col] - 1;
        for (unsigned e = csr_diag_ptr[col] + 1; e < csr_r_ptr[col + 1]; ++e) {
            const unsigned pivot = csr_pivot_pos[e], target = csr_c_idx[e];
            if (pivot == ~0u) return;
            upper_bound += std::min(lower, sym_c_ptr[target + 1] - pivot - 1);
            if (upper_bound > limit) return;
        }
    }
    destination_lower.reserve(upper_bound); destination_upper.reserve(upper_bound);
    destination_level_ptr.assign(num_lev + 1, 0); destination_group_ptr.push_back(0);
    struct Product { unsigned target, lower, upper; };
    std::vector<Product> products;
    for (int lv = 0; lv < num_lev; ++lv) {
        products.clear();
        for (int pos = level_ptr[lv]; pos < level_ptr[lv + 1]; ++pos) {
            const unsigned col = level_idx[pos];
            for (unsigned e = csr_diag_ptr[col] + 1; e < csr_r_ptr[col + 1]; ++e) {
                const unsigned target = csr_c_idx[e], coefficient = csr_pivot_pos[e];
                unsigned src = l_col_ptr[col] + 1, dst = coefficient + 1;
                const unsigned se = sym_c_ptr[col + 1], de = sym_c_ptr[target + 1];
                const auto emit = [&](unsigned a, unsigned b) { products.push_back(Product{b, a, coefficient}); };
                if (size_t(de - dst) > size_t(se - src) * 32) {
                    for (; src < se && dst < de; ++src) {
                        dst = std::lower_bound(sym_r_idx.begin() + dst, sym_r_idx.begin() + de, sym_r_idx[src]) - sym_r_idx.begin();
                        if (dst < de && sym_r_idx[dst] == sym_r_idx[src]) emit(src, dst++);
                    }
                } else if (size_t(se - src) > size_t(de - dst) * 32) {
                    for (; dst < de && src < se; ++dst) {
                        src = std::lower_bound(sym_r_idx.begin() + src, sym_r_idx.begin() + se, sym_r_idx[dst]) - sym_r_idx.begin();
                        if (src < se && sym_r_idx[src] == sym_r_idx[dst]) emit(src++, dst);
                    }
                } else while (src < se && dst < de) {
                    if (sym_r_idx[src] < sym_r_idx[dst]) ++src;
                    else if (sym_r_idx[dst] < sym_r_idx[src]) ++dst;
                    else { emit(src, dst); ++src; ++dst; }
                }
            }
        }
        std::sort(products.begin(), products.end(), [](const Product &a, const Product &b) {
            return a.target < b.target || (a.target == b.target && a.upper < b.upper);
        });
        for (size_t first = 0; first < products.size();) {
            size_t end = first + 1;
            while (end < products.size() && products[end].target == products[first].target) ++end;
            destination_target.push_back(products[first].target);
            destination_max_fanin = std::max(destination_max_fanin, static_cast<unsigned>(end - first));
            for (size_t p = first; p < end; ++p) {
                destination_lower.push_back(products[p].lower); destination_upper.push_back(products[p].upper);
            }
            destination_group_ptr.push_back(destination_lower.size()); first = end;
        }
        destination_level_ptr[lv + 1] = destination_target.size();
    }
    destination_products = destination_lower.size(); destination_groups = destination_target.size();
    if (destination_products < 1024u * 1024u ||
        size_t(destination_products) * 4 < size_t(destination_groups) * 5) {
        destination_level_ptr.clear(); destination_group_ptr.clear(); destination_target.clear();
        destination_lower.clear(); destination_upper.clear(); return;
    }
    destination_ready = true;
}

void Symbolic_Matrix :: leveling()
{
    vector<int> inlevel, level_size(n, 0);
    inlevel.reserve(n);
    size_t left_edges = 0, omitted_left_edges = 0;

    for (unsigned i = 0; i < n; ++i)
    {
        int max_lv = -1, lv;
        //search dependent columns on the left
        for (unsigned j = sym_c_ptr[i]; j < l_col_ptr[i]; ++j) {
            unsigned nz_idx = sym_r_idx[j]; //Nonzero row in col i, U part

            //L part of col nz_idx exists , U-dependency found
            if (l_col_ptr[nz_idx] + 1 != sym_c_ptr[nz_idx+1]) {
                lv = inlevel[nz_idx];
                if (lv > max_lv)
                    max_lv = lv;
            }
        }
        for (unsigned j = csr_r_ptr[i]; j < csr_diag_ptr[i]; ++j) {
            unsigned nz_idx = csr_c_idx[j];
            ++left_edges;
            // An extra left edge protects a future-U coefficient read.
            // No target beyond i means no such write; own-column writes
            // remain ordered by the unchanged U-dependency scan above.
            const unsigned predecessor_end = csr_r_ptr[nz_idx + 1];
            if (csr_diag_ptr[nz_idx] + 1 == predecessor_end ||
                csr_c_idx[predecessor_end - 1] <= i) {
                ++omitted_left_edges;
                continue;
            }
            lv = inlevel[nz_idx];
            if (lv > max_lv)
                max_lv = lv;
        }        
        lv = max_lv + 1;
        inlevel.push_back(lv);
        ++level_size[lv];
        if (lv > num_lev)
            num_lev = lv;
    }

    ++num_lev;

    level_ptr.reserve(num_lev);
    level_ptr.push_back(0);
    for (int i = 0; i < num_lev; ++i)
        level_ptr.push_back(level_ptr[i] + level_size[i]);

    level_idx.resize(n);
    vector<int> tlen(level_ptr);
    for (unsigned i = 0; i < n; ++i)
        level_idx[tlen[inlevel[i]]++] = i;

    prepare_dense_tail(inlevel);
    prepare_near_dense_tail(inlevel);

    // Structure-only tasks, in level/source order. Heavy narrow levels may
    // share the task backend; cheap narrow levels retain the fused kernel.
    batch_task_ptr.assign(n + 1, 0);
    size_t empty_tasks_skipped = 0;
    for (int lv = 0; lv < num_lev; ++lv) {
        const size_t level_task_head = batch_task_entry.size();
        const unsigned count = level_ptr[lv + 1] - level_ptr[lv];
        size_t targets = 0;
        bool narrow_heavy = false;
        for (int pos = level_ptr[lv]; pos < level_ptr[lv + 1]; ++pos) {
            const unsigned col = level_idx[pos];
            const unsigned upper = csr_r_ptr[col + 1] - csr_diag_ptr[col] - 1;
            const unsigned lower = sym_c_ptr[col + 1] - l_col_ptr[col] - 1;
            targets += upper;
            if (upper > 64 || lower > 2048) narrow_heavy = true;
        }
        const bool split = count > 448 ||
            ((count > 16 || narrow_heavy) && targets >= size_t(32) * count);
        for (int pos = level_ptr[lv]; pos < level_ptr[lv + 1]; ++pos) {
            const unsigned col = level_idx[pos];
            if (split) for (unsigned entry = csr_diag_ptr[col] + 1;
                           entry < csr_r_ptr[col + 1]; ++entry) {
                const unsigned target = csr_c_idx[entry];
                const unsigned pivot = csr_pivot_pos[entry];
                const unsigned end = sym_c_ptr[target + 1];
                const unsigned lower = sym_c_ptr[col + 1] - l_col_ptr[col] - 1;
                const unsigned tail = pivot == ~0u ? 0 : end - pivot - 1;
                const bool tile = pivot != ~0u && tail > 256 &&
                    !(lower > 0 && tail / 32u > lower);
                for (unsigned first = tile ? pivot + 1 : 0;; first += 256) {
                    const unsigned last = tile ? std::min(first + 256u, end) : end;
                    bool empty = lower == 0;
                    if (tile && !empty) {
                        const auto source_end = sym_r_idx.begin() + sym_c_ptr[col + 1];
                        const auto hit = std::lower_bound(
                            sym_r_idx.begin() + l_col_ptr[col] + 1,
                            source_end, sym_r_idx[first]);
                        empty = hit == source_end || *hit > sym_r_idx[last - 1];
                    }
                    // Preserve the original split backend even if an entire
                    // level is empty. Source preparation remains unchanged.
                    if (!empty || batch_task_entry.size() == level_task_head) {
                        batch_task_entry.push_back(entry);
                        batch_task_source.push_back(pos);
                        batch_task_begin.push_back(tile ? first : ~0u);
                        batch_task_end.push_back(tile ? last : ~0u);
                    } else {
                        ++empty_tasks_skipped;
                    }
                    if (!tile || first + 256 >= end) break;
                }
            }
            batch_task_ptr[pos + 1] = batch_task_entry.size();
        }
    }
    std::cout << "Structurally empty update tasks skipped: " << empty_tasks_skipped << std::endl;
    // Cache index intersections only: no numerical values or factorization.
    const size_t pair_limit = 8u * 1024u * 1024u;
    std::vector<unsigned char> unique_column(n, 1);
    for (unsigned col = 0; col < n; ++col)
        for (unsigned p = sym_c_ptr[col] + 1; p < sym_c_ptr[col + 1]; ++p)
            if (sym_r_idx[p] == sym_r_idx[p - 1]) unique_column[col] = 0;
    unsigned covered_tasks = 0;
    for (size_t task = 0; task < batch_task_entry.size(); ++task) {
        const unsigned col = level_idx[batch_task_source[task]];
        const unsigned entry = batch_task_entry[task];
        const unsigned target = csr_c_idx[entry];
        const unsigned pivot = csr_pivot_pos[entry];
        update_pair_begin.push_back(~0u);
        update_pair_end.push_back(~0u);
        if (!unique_column[col] || pivot == ~0u || update_pair_source.size() >= pair_limit) continue;
        unsigned dst = batch_task_begin[task] == ~0u ? pivot + 1 : batch_task_begin[task];
        const unsigned end = batch_task_begin[task] == ~0u ? sym_c_ptr[target + 1] : batch_task_end[task];
        unsigned src = l_col_ptr[col] + 1;
        const unsigned source_end = sym_c_ptr[col + 1];
        if (dst < end) src = std::lower_bound(sym_r_idx.begin() + src,
            sym_r_idx.begin() + source_end, sym_r_idx[dst]) - sym_r_idx.begin();
        const size_t first = update_pair_source.size();
        bool over_budget = false;
        if (src < source_end && (end - dst) / 32u > source_end - src) {
            for (; src < source_end && dst < end; ++src) {
                dst = std::lower_bound(sym_r_idx.begin() + dst,
                    sym_r_idx.begin() + end, sym_r_idx[src]) - sym_r_idx.begin();
                if (dst < end && sym_r_idx[dst] == sym_r_idx[src]) {
                    if (update_pair_source.size() == pair_limit) { over_budget = true; break; }
                    update_pair_source.push_back(src);
                    update_pair_target.push_back(dst++);
                }
            }
        } else {
            while (src < source_end && dst < end) {
                const unsigned sr = sym_r_idx[src], dr = sym_r_idx[dst];
                if (sr < dr) { ++src; continue; }
                if (dr < sr) { ++dst; continue; }
                if (update_pair_source.size() == pair_limit) { over_budget = true; break; }
                update_pair_source.push_back(src++);
                update_pair_target.push_back(dst++);
            }
        }
        if (over_budget) {
            update_pair_source.resize(first);
            update_pair_target.resize(first);
        } else {
            update_pair_begin[task] = first;
            update_pair_end[task] = update_pair_source.size();
            ++covered_tasks;
        }
    }
    // Queue only structural task identifiers, in original source order.
    short_update_task_ptr.assign(n + 1, 0);
    for (unsigned pos = 0; pos < n; ++pos) {
        for (unsigned task = batch_task_ptr[pos]; task < batch_task_ptr[pos + 1]; ++task)
            if (update_pair_begin[task] != ~0u &&
                update_pair_end[task] - update_pair_begin[task] <= 16)
                short_update_task_index.push_back(task);
        short_update_task_ptr[pos + 1] = short_update_task_index.size();
    }
    // Source positions only. No numerical normalization is performed here.
    cached_prepare_ptr.assign(n + 1, 0);
    scratch_prepare_ptr.assign(n + 1, 0);
    size_t zero_prepare_sources = 0;
    for (int lv = 0; lv < num_lev; ++lv) {
        const unsigned first = level_ptr[lv], end = level_ptr[lv + 1];
        const bool active = batch_task_ptr[end] > batch_task_ptr[first] &&
            (dense_columns.empty() || lv < dense_first_level || lv >= dense_end_level);
        for (unsigned pos = first; pos < end; ++pos) {
            const unsigned col = level_idx[pos];
            const unsigned lower = sym_c_ptr[col + 1] - l_col_ptr[col] - 1;
            if (active && lower) {
                bool needs_scratch = lower > 1024;
                for (unsigned task = batch_task_ptr[pos]; task < batch_task_ptr[pos + 1]; ++task)
                    if (update_pair_begin[task] == ~0u) needs_scratch = true;
                if (needs_scratch) scratch_prepare_sources.push_back(pos);
                else cached_prepare_sources.push_back(pos);
            } else if (active) ++zero_prepare_sources;
            cached_prepare_ptr[pos + 1] = cached_prepare_sources.size();
            scratch_prepare_ptr[pos + 1] = scratch_prepare_sources.size();
        }
    }
    m_out << "Source preparation queues: cached " << cached_prepare_sources.size()
          << ", scratch " << scratch_prepare_sources.size()
          << ", zero lower omitted " << zero_prepare_sources << endl;
    m_out << "Cached short task queue: " << short_update_task_index.size() << endl;
    m_out << "Cached intersection tasks: " << covered_tasks << "/" << batch_task_entry.size()
          << ", pairs: " << update_pair_source.size() << endl;
    prepare_destination_groups();
    m_out << "Destination group plan: enabled " << destination_ready << ", products " << destination_products
          << ", groups " << destination_groups << ", max fanin " << destination_max_fanin << endl;
    m_out << "Number of levels: " << num_lev << endl;
    m_out << "Extra left edges removed: " << omitted_left_edges << "/" << left_edges << endl;
}

#if GLU_DEBUG
void Symbolic_Matrix::PrintLevel()
{
    for (int i = 0; i < num_lev; ++i) {
        for (int j = level_ptr[i]; j < level_ptr[i + 1]; ++j)
            m_out << level_idx[j] << ' ';
        m_out << '\n';
    }
}

void Symbolic_Matrix::ABFTCalculateCCA()
{
    CCA.assign(n, 0.0);
    for (unsigned i = 0; i < n; ++i) {
        for (unsigned j = sym_c_ptr[i]; j < sym_c_ptr[i + 1]; ++j)
            CCA[i] += val[j];
    }
}

void Symbolic_Matrix::ABFTCheckResult()
{
     vector<REAL> CCL(n, 1.0);
     for(unsigned i = 0; i < n; i++)
        for(unsigned j = l_col_ptr[i] + 1; j < sym_c_ptr[i + 1]; j++)
            CCL[i] += val[j];

     vector<REAL> CCA_ABFT(n, 0.0);
     for(unsigned i = 0; i < n; i++)
     {
        CCA_ABFT[i] = 0;
        for(unsigned j = sym_c_ptr[i]; j <= l_col_ptr[i]; j++)
            CCA_ABFT[i] += CCL[sym_r_idx[j]] * val[j];
     }

     //Compare CCA and CCA_ABFT
     for(unsigned i = 0; i < n; ++i) {
         if (abs(CCA_ABFT[i] - CCA[i]) > 1e-5) {
             m_err << "Column " << i << ": CCA = " << CCA[i] <<
                 ", CCA_ABFT[i] = " << CCA_ABFT[i] << endl;
             m_err << "More to come..." << endl;
             return;
         }
     }
     m_out << "Results passed ABFT check." << endl;
}
#endif

vector<REAL> Symbolic_Matrix::solve(SNicsLU *nicslu, const vector<REAL> &rhs)
{
    vector<REAL> b(n);
    vector<REAL> x(n);
    if (n == 0)
        return x;

    unsigned mc64_scale = nicslu->cfgi[1];

    unsigned *rp = nicslu->col_perm;
    unsigned *cp = nicslu->row_perm_inv;
    int *piv = nicslu->pivot;
    double *rows = nicslu->col_scale_perm;
    double *cols = nicslu->row_scale;

    //apply row permutation and row scaling to rhs
    if (mc64_scale)
        for (unsigned j = 0; j < n; ++j) {
            unsigned p = piv[j];
            b[j] = rhs[rp[piv[j]]] * rows[p];
        }
    else
        for (unsigned j = 0; j < n; ++j)
            b[j] = rhs[rp[piv[j]]];

    //left-multiply inv(L)
    for (unsigned j = 0; j < n; ++j) {
        for (unsigned p = l_col_ptr[j] + 1; p < sym_c_ptr[j + 1]; ++p)
            b[sym_r_idx[p]] -= val[p] * b[j];
    }

    //left-multiply inv(U)
    for (int jj = n - 1; jj >= 0; --jj) {
        unsigned diag = l_col_ptr[jj];
        b[jj] /= val[diag];

        for (unsigned p = sym_c_ptr[jj]; p < l_col_ptr[jj]; ++p) {
            b[sym_r_idx[p]] -= val[p] * b[jj];
        }
    }

    //apply col permutation and scaling
    if (mc64_scale)
        for (unsigned j = 0; j < n; ++j)
            x[j] = b[cp[j]] * cols[j];
    else
        for (unsigned j = 0; j < n; ++j)
            x[j] = b[cp[j]];


    return x;
}

static inline REAL triangular_dot(const REAL* __restrict__ values,
                                  const unsigned* __restrict__ indices,
                                  const REAL* __restrict__ rhs,
                                  unsigned begin, unsigned end) {
    REAL sum = 0;
    unsigned p = begin;
    if (end - begin >= 16) {
        REAL s0 = 0, s1 = 0, s2 = 0, s3 = 0;
        for (; end - p >= 4; p += 4) {
            s0 += values[p] * rhs[indices[p]];
            s1 += values[p + 1] * rhs[indices[p + 1]];
            s2 += values[p + 2] * rhs[indices[p + 2]];
            s3 += values[p + 3] * rhs[indices[p + 3]];
        }
        sum = (s0 + s1) + (s2 + s3);
    }
    for (; p < end; ++p) sum += values[p] * rhs[indices[p]];
    return sum;
}

vector<REAL> Symbolic_Matrix::solve_CSR(SNicsLU *nicslu, const vector<REAL> &rhs)
{
    vector<REAL> b(n);
    vector<REAL> x(n);
    if (n == 0)
        return x;

    unsigned mc64_scale = nicslu->cfgi[1];

    unsigned *rp = nicslu->row_perm;
    unsigned *cp = nicslu->col_perm_inv;
    int *piv = nicslu->pivot_inv;
    double *rows = nicslu->row_scale;
    double *cols = nicslu->col_scale_perm;

    // Apply row permutation and scaling to rhs
    if (mc64_scale) {
        for (unsigned j = 0; j < n; ++j) {
            unsigned p = rp[j];
            b[j] = rhs[p] * rows[p];
        }
    } else {
        for (unsigned j = 0; j < n; ++j) {
            b[j] = rhs[rp[j]];
        }
    }

    // Left-multiply inv(L) - lower triangular solve with diagonal division
    for (unsigned j = 0; j < n; ++j) {
        REAL sum = 0.0;
        unsigned diag = l_col_ptr[j];

        sum = triangular_dot(val.data(), sym_r_idx.data(), b.data(), sym_c_ptr[j], l_col_ptr[j]);
        b[j] = (b[j] - sum) / val[diag];
    }

    // Left-multiply inv(U) - upper triangular solve (no division, U diag already applied)
    for (int jj = n - 1; jj >= 0; --jj) {
        REAL sum = 0.0;

        sum = triangular_dot(val.data(), sym_r_idx.data(), b.data(), l_col_ptr[jj] + 1, sym_c_ptr[jj + 1]);

        b[jj] -= sum;
    }

    // Apply column permutation and scaling
    if (mc64_scale) {
        for (unsigned j = 0; j < n; ++j) {
            unsigned p = cp[j];
            x[j] = b[piv[p]] * cols[p];
        }
    } else {
        for (unsigned j = 0; j < n; ++j) {
            x[j] = b[piv[cp[j]]];
        }
    }

    return x;
}
