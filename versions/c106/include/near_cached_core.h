// Research helper: cache one closed forward component and test sorted suffixes.
// Returns conservative candidates; does not claim exhaustive near-dense coverage.
#include <algorithm>
#include <vector>

struct NearCandidate {
    std::vector<unsigned> columns;
    int head = -1, end = -1;
    size_t entries = 0;
};

struct NearCacheStats {
    size_t edge_visits = 0;
    unsigned roots = 0, components = 0, overflow = 0, proven = 0;
    bool budget_exhausted = false;
};

template<class Accept>
void scan_near_suffixes(const Symbolic_Matrix &a, const std::vector<unsigned> &levels,
                       const std::vector<unsigned> &lower, const std::vector<unsigned> &upper,
                       NearCacheStats &stats, Accept accept) {
    const size_t edge_budget = 8000000;
    std::vector<unsigned> position(a.n, ~0u), active_prefix(a.num_lev+1, 0), component;
    for (int lv = 0; lv < a.num_lev; ++lv) {
        active_prefix[lv+1] = active_prefix[lv];
        for (int p = a.level_ptr[lv]; p < a.level_ptr[lv+1]; ++p)
            active_prefix[lv+1] += lower[a.level_idx[p]] != 0;
    }
    std::vector<size_t> entries;
    std::vector<unsigned> active, minimum_level;
    std::vector<int> last_level, outside;
    unsigned tree_base = 0;
    bool cache_valid = false;
    const auto visit = [&]() {
        if (++stats.edge_visits <= edge_budget) return true;
        stats.budget_exhausted = true;
        return false;
    };
    for (int head = 1; head < a.num_lev && !stats.budget_exhausted; ++head) {
        unsigned root = ~0u, count = 0;
        for (int p = a.level_ptr[head]; p < a.level_ptr[head+1]; ++p)
            if (lower[a.level_idx[p]]) { root = a.level_idx[p]; ++count; }
        if (count != 1 || size_t(lower[root]) + upper[root] + 1 < 256) continue;
        ++stats.roots;
        if (!cache_valid || position[root] == ~0u) {
            for (unsigned c : component) position[c] = ~0u;
            component.assign(1, root);
            position[root] = 0;
            cache_valid = false;
            ++stats.components;
            bool ok = true;
            const auto add = [&](unsigned c) {
                if (position[c] == ~0u) {
                    position[c] = component.size();
                    component.push_back(c);
                }
            };
            for (size_t i = 0; ok && i < component.size(); ++i) {
                const unsigned c = component[i];
                for (unsigned p = a.l_col_ptr[c]+1; p < a.sym_c_ptr[c+1]; ++p) {
                    if (!visit()) { ok = false; break; }
                    add(a.sym_r_idx[p]);
                    if (component.size() > 4096) { ok = false; break; }
                }
                for (unsigned p = a.csr_diag_ptr[c]+1; ok && p < a.csr_r_ptr[c+1]; ++p) {
                    if (!visit()) { ok = false; break; }
                    add(a.csr_c_idx[p]);
                    if (component.size() > 4096) { ok = false; break; }
                }
            }
            if (!ok) { ++stats.overflow; continue; }
            std::sort(component.begin(), component.end());
            const unsigned size = component.size();
            for (unsigned i = 0; i < size; ++i) position[component[i]] = i;
            entries.assign(size+1, 0);
            active.assign(size+1, 0);
            minimum_level.assign(size+1, a.num_lev);
            last_level.assign(size+1, -1);
            for (unsigned i = size; i-- > 0;) {
                const unsigned c = component[i];
                entries[i] = entries[i+1] + 1u + size_t(lower[c]) + upper[c];
                active[i] = active[i+1] + (lower[c] != 0);
                minimum_level[i] = std::min(minimum_level[i+1], levels[c]);
                last_level[i] = std::max(last_level[i+1], lower[c] ? int(levels[c]) : -1);
            }
            tree_base = 1;
            while (tree_base < size) tree_base *= 2;
            outside.assign(tree_base*2, -1);
            // A predecessor edge is outside suffix cut iff dep_pos < cut <= owner_pos.
            // Outside-component predecessors cover every cut <= owner_pos.
            const auto dependency = [&](unsigned dep, unsigned owner) {
                unsigned lo = position[dep] == ~0u ? 0 : position[dep]+1;
                unsigned hi = owner+1;
                for (lo += tree_base, hi += tree_base; lo < hi; lo /= 2, hi /= 2) {
                    if (lo & 1u) { outside[lo] = std::max(outside[lo], int(levels[dep])); ++lo; }
                    if (hi & 1u) { --hi; outside[hi] = std::max(outside[hi], int(levels[dep])); }
                }
            };
            for (unsigned i = 0; ok && i < size; ++i) {
                const unsigned c = component[i];
                for (unsigned p = a.sym_c_ptr[c]; p < a.l_col_ptr[c]; ++p) {
                    if (!visit()) { ok = false; break; }
                    const unsigned dep = a.sym_r_idx[p];
                    if (lower[dep]) dependency(dep, i);
                }
                for (unsigned p = a.csr_r_ptr[c]; ok && p < a.csr_diag_ptr[c]; ++p) {
                    if (!visit()) { ok = false; break; }
                    dependency(a.csr_c_idx[p], i);
                }
            }
            if (!ok) continue;
            for (unsigned i = 1; i < tree_base; ++i) {
                outside[2*i] = std::max(outside[2*i], outside[i]);
                outside[2*i+1] = std::max(outside[2*i+1], outside[i]);
            }
            cache_valid = true;
        }
        const unsigned cut = position[root];
        const unsigned size = component.size() - cut;
        if (size < 512 || size > 4096 || entries[cut] > size_t(size)*size) continue;
        const int end = last_level[cut];
        if (minimum_level[cut] < unsigned(head) || end < head || outside[tree_base+cut] >= head) continue;
        if (active_prefix[end+1] - active_prefix[head] != active[cut]) continue;
        ++stats.proven;
        accept(component, cut, entries[cut], head, end+1);
    }
}
