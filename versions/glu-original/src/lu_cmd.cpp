#include <iostream>
#include <vector>
#include <set>
#include <iomanip>
#include <cstring>
#include <cmath>
#include <fstream>
#include <chrono>
#include "symbolic.h"
#include "numeric.h"
#include "Timer.h"
#include "preprocess.h"
#include "nicslu.h"

using namespace std;

void help_message()
{
    cout << endl;
    cout << "GLU program V3.0" << endl;
    cout << "Usage: ./lu_cmd -i inputfile" << endl;
    cout << "Additional usage: ./lu_cmd -i inputfile -p" << endl;
    cout << "-p to enable perturbation" << endl;
}

int main(int argc, char** argv)
{
    PhaseReport report;
    const auto phase_ms = [&]() { return report.now(); };
    Timer t;
    double utime;
    SNicsLU *nicslu = nullptr;

    const char *matrixName = nullptr;
    bool PERTURB = false;

    double *ax = NULL, *ax_backup = NULL;
    unsigned int *ai = NULL, *ai_backup = NULL;
    unsigned int *ap = NULL, *ap_backup = NULL;
    unsigned int n, nnz_backup;

    if (argc < 3) {
        help_message();
        return -1;
    }

    for (int i = 1; i < argc;) {
        if (strcmp(argv[i], "-i") == 0) {
            if(i + 1 >= argc) {
                help_message();
                return -1;
            }
            matrixName = argv[i+1];
            i += 2;
        }
        else if (strcmp(argv[i], "-p") == 0) {
            PERTURB = true;
            i += 1;
        }        
        else {
            help_message();
            return -1;
        }
    }

    if (matrixName == nullptr) {
        help_message();
        return -1;
    }

    nicslu = static_cast<SNicsLU *>(malloc(sizeof(SNicsLU)));
    if (nicslu == nullptr) {
        cerr << "Failed to allocate NicsLU context." << endl;
        return -1;
    }

    // Build a permutation/scaling-adjusted matrix and keep the numeric arrays in CSC.
    const double preprocessing_start = phase_ms();
    int err = preprocess(matrixName, nicslu, &ax, &ai, &ap, &ax_backup, &ai_backup, &ap_backup, &nnz_backup);
    if (err)
    {
        cerr << "Matrix preprocessing failed." << endl;
        free(nicslu);
        return -1;
    }

    report.add("preprocessing", preprocessing_start, true);
    n = nicslu->n;

    cout << "Matrix Row: " << n << endl;
    cout << "Original nonzero: " << nicslu->nnz << endl;

    t.start();

    // Symbolic analysis: fill-in prediction, CSR transpose, and level scheduling.
    const double symbolic_start = phase_ms();
    Symbolic_Matrix A_sym(n, cout, cerr);
    A_sym.fill_in(ai, ap);
    report.add("symbolic_analysis", symbolic_start, true);
    t.elapsedUserTime(utime);
    cout << "Symbolic time: " << utime << " ms" << endl;

    t.start();
    const double csr_start = phase_ms();
    A_sym.csr();
    report.add("structure_preparation", csr_start, true);
    t.elapsedUserTime(utime);
    cout << "CSR time: " << utime << " ms" << endl;

    t.start();
    const double numeric_prepare_start = phase_ms();
    A_sym.predictLU(ai, ap, ax);
    report.add("numeric_preparation", numeric_prepare_start, false);
    t.elapsedUserTime(utime);
    cout << "PredictLU time: " << utime << " ms" << endl;

    t.start();
    const double leveling_start = phase_ms();
    A_sym.leveling();
    report.add("structure_preparation", leveling_start, true);
    t.elapsedUserTime(utime);
    cout << "Leveling time: " << utime << " ms" << endl;

#if GLU_DEBUG
    A_sym.ABFTCalculateCCA();
//    A_sym.PrintLevel();
#endif

    // Numeric factorization on GPU updates A_sym.val in place to LU factors.
    LUonDevice(A_sym, cout, cerr, PERTURB, &report);

#if GLU_DEBUG
    A_sym.ABFTCheckResult();
#endif

    // Solve Ax=b with the computed factors and NICSLU permutations/scales.
    const double solve_start = phase_ms();
    vector<REAL> b(n, 1.);
    vector<REAL> x = A_sym.solve_CSR(nicslu, b);
    report.add("triangular_solve", solve_start, false);
    {
        ofstream x_f("x.dat");
        x_f << std::scientific << std::setprecision(17);
        for (REAL xx: x)
            x_f << xx << '\n';
    }

    // Residual verification: compute ||Ax - b|| using the original matrix
    {
        REAL n1 = 0., n2 = 0., ni = 0.;
        for (unsigned i = 0; i < n; ++i) {
            REAL sum = 0.;
            unsigned end = ap_backup[i + 1];
            for (unsigned j = ap_backup[i]; j < end; ++j) {
                sum += (REAL)ax_backup[j] * x[ai_backup[j]];
            }
            REAL tmp = sum - b[i];
            if (tmp < 0.) tmp = -tmp;
            n1 += tmp;
            n2 += tmp * tmp;
            if (tmp > ni) ni = tmp;
        }
        cout << "Ax-b (1-norm): " << n1 << endl;
        cout << "Ax-b (2-norm): " << sqrt(n2) << endl;
        cout << "Ax-b (infinite-norm): " << ni << endl;
    }

    NicsLU_Destroy(nicslu);
    free(nicslu);
    free(ax);
    free(ai);
    free(ap);
    free(ax_backup);
    free(ai_backup);
    free(ap_backup);

    return report.write() ? 0 : 1;
}
