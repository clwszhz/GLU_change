#ifndef CASE7_PHASE_REPORT_H
#define CASE7_PHASE_REPORT_H
#include <chrono>
#include <fstream>
#include <iomanip>
#include <vector>
struct PhaseReport {
    struct Phase { const char *name; double start, end; bool reusable; };
    std::chrono::steady_clock::time_point origin = std::chrono::steady_clock::now();
    std::vector<Phase> phases;
    double now() const { return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-origin).count(); }
    void add(const char *name, double start, bool reusable) { phases.push_back({name,start,now(),reusable}); }
    bool write() const {
        std::ofstream out("timings.json");
        out << std::setprecision(17) << "{\"schema\":\"case7-phases-v2\",\"clock\":\"monotonic_wall\",\"unit\":\"ms\",\"gpu_synchronized\":true,\"measured_total_ms\":" << now();
        for (int category=0; category<2; ++category) {
            out << (category==0 ? ",\"reusable_phases\":[" : ",\"diagnostic_phases\":[");
            bool first=true;
            for (const auto &p:phases) if (p.reusable == (category==0)) {
                if (!first) out << ',';
                first=false;
                out << "{\"name\":\"" << p.name << "\",\"start_ms\":" << p.start << ",\"end_ms\":" << p.end << '}';
            }
            out << ']';
        }
        out << "}\n"; out.close(); return bool(out);
    }
};
#endif
