// =============================================================================
//  Minimal dependency-free test harness.
//  Phase 1 stays dependency-free so the suite builds & verifies offline
//  (GoogleTest/Benchmark arrive with the network-enabled Phase 2 build).
// =============================================================================
#ifndef STRATA_TEST_HARNESS_HPP
#define STRATA_TEST_HARNESS_HPP

#include <cstdio>
#include <exception>
#include <string>
#include <vector>

namespace strata_test {

struct Case { const char* name; void (*fn)(); };

inline std::vector<Case>& registry() {
    static std::vector<Case> r;
    return r;
}

inline int& failures() { static int f = 0; return f; }

struct Registrar {
    Registrar(const char* name, void (*fn)()) { registry().push_back({name, fn}); }
};

#define STRATA_TEST(name)                                                      \
    static void name();                                                        \
    static ::strata_test::Registrar reg_##name{#name, &name};                  \
    static void name()

#define STRATA_CHECK(cond)                                                     \
    do {                                                                       \
        if (!(cond)) {                                                         \
            ++::strata_test::failures();                                       \
            std::printf("    FAIL %s:%d  CHECK(%s)\n",                         \
                        __FILE__, __LINE__, #cond);                            \
        }                                                                      \
    } while (0)

#define STRATA_CHECK_EQ(a, b)                                                  \
    do {                                                                       \
        auto _va = (a); auto _vb = (b);                                        \
        if (!(_va == _vb)) {                                                   \
            ++::strata_test::failures();                                       \
            std::printf("    FAIL %s:%d  CHECK_EQ(%s, %s)\n",                  \
                        __FILE__, __LINE__, #a, #b);                           \
        }                                                                      \
    } while (0)

#define STRATA_CHECK_THROWS(expr, ex)                                          \
    do {                                                                       \
        bool _caught = false;                                                  \
        try { (void)(expr); } catch (const ex&) { _caught = true; }            \
        catch (...) {}                                                         \
        if (!_caught) {                                                        \
            ++::strata_test::failures();                                       \
            std::printf("    FAIL %s:%d  expected %s\n",                       \
                        __FILE__, __LINE__, #ex);                              \
        }                                                                      \
    } while (0)

inline int run_all() {
    int passed = 0;
    for (const auto& c : registry()) {
        const int before = failures();
        std::printf("[ RUN  ] %s\n", c.name);
        try {
            c.fn();
        } catch (const std::exception& e) {
            ++failures();
            std::printf("    FAIL unexpected exception: %s\n", e.what());
        }
        if (failures() == before) { ++passed; std::printf("[  OK  ] %s\n", c.name); }
        else                      { std::printf("[ FAIL ] %s\n", c.name); }
    }
    std::printf("\n%d/%zu cases passed, %d check failure(s)\n",
                passed, registry().size(), failures());
    return failures() == 0 ? 0 : 1;
}

} // namespace strata_test

#endif // STRATA_TEST_HARNESS_HPP
