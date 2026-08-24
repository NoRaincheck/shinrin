/*
 * C ABI bridge between shinrin's native extension (PyO3/Rust) and the
 * vendored SPOTSET engine — treeFARMS ("Exploring the Whole Rashomon Set of
 * Sparse Decision Trees", ubc-systopia/treeFARMS, BSD-3-Clause), renamed
 * SPOTSET (Sparse Optimal Rashomon Trees) in this project. All engine
 * symbols live in namespace spotset so both this and the SPOT engine can be
 * linked into one shared library.
 *
 * Mirrors upstream's pybind11 module (libgosdt): configuration arrives as a
 * JSON string and training data as a CSV string (with header; last column is
 * the label). The engine's own Encoder binarizes rational/integral/categorical
 * columns, so no Python-side binarization is required or performed.
 *
 * Note: like SPOT, the engine unconditionally includes <gmp.h>, so it links
 * against the bundled mini-gmp + mpn shim regardless of
 * SHINRIN_CORELS_NO_GMP.
 */

#include <cstring>
#include <exception>
#include <new>
#include <sstream>
#include <string>

#include "engine/gosdt.hpp"

extern "C" {

struct ShinrinSpotsetResult {
    char *model_json;      /* JSON array describing every extracted tree */
    float time_elapsed;    /* seconds spent training (engine static) */
    unsigned int graph_size;
    unsigned int n_iterations;
    int status;            /* engine status code: 0=CONVERGED, 2=TIMEOUT, ... */
};

static char *dup_c_string(char const *s) {
    if (s == NULL) {
        return NULL;
    }
    size_t n = strlen(s);
    char *out = static_cast<char *>(malloc(n + 1));
    if (out != NULL) {
        memcpy(out, s, n + 1);
    }
    return out;
}

/*
 * Configure the engine from a JSON object string. Returns 0 on success,
 * 1 on a caught C++ exception (*error_message populated), -1 on allocation
 * failure.
 */
int shinrin_spotset_configure(const char *config_json, char **error_message) {
    try {
        std::istringstream stream(config_json != NULL ? config_json : "{}");
        spotset::GOSDT::configure(stream);
        return 0;
    } catch (std::exception const &e) {
        *error_message = dup_c_string(e.what());
        return *error_message != NULL ? 1 : -1;
    } catch (...) {
        *error_message = dup_c_string("unknown SPOTSET exception");
        return *error_message != NULL ? 1 : -1;
    }
}

/*
 * Fit a Rashomon set from CSV data (header row required, label is the last
 * column). On success (*out populated, model_json must be freed by the
 * caller via shinrin_spotset_free), returns 0. Returns 1 on a caught C++
 * exception (*error_message populated), -1 on allocation failure.
 */
int shinrin_spotset_fit(const char *dataset_csv, ShinrinSpotsetResult *out,
                        char **error_message) {
    try {
        spotset::GOSDT model;
        std::istringstream stream(dataset_csv != NULL ? dataset_csv : "");
        std::string result;
        model.fit(stream, result);

        out->model_json = dup_c_string(result.c_str());
        out->time_elapsed = spotset::GOSDT::time;
        out->graph_size = spotset::GOSDT::size;
        out->n_iterations = spotset::GOSDT::iterations;
        out->status = static_cast<int>(spotset::GOSDT::status);

        return out->model_json != NULL ? 0 : -1;
    } catch (std::exception const &e) {
        *error_message = dup_c_string(e.what());
        return *error_message != NULL ? 1 : -1;
    } catch (...) {
        *error_message = dup_c_string("unknown SPOTSET exception");
        return *error_message != NULL ? 1 : -1;
    }
}

void shinrin_spotset_free(void *p) { free(p); }

}  // extern "C"
