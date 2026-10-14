#ifndef _NUMERIC_H_
#define _NUMERIC_H_
#include "symbolic.h"
#include "phase_report.h"

using namespace std;
void LUonDevice(Symbolic_Matrix &, ostream &, ostream &, bool, PhaseReport * = nullptr);

#endif
