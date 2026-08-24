/*
 * C ABI bridge between shinrin's native extension (PyO3/Rust) and the
 * vendored SPOT engine — gosdt-guesses ("Fast Sparse Decision Tree
 * Optimization via Reference Ensembles", ubc-systopia/gosdt-guesses,
 * BSD-3-Clause), renamed SPOT (SParse OpTimal) in this project.
 *
 * Replaces upstream's pybind11 module (_libgosdt):
 *   - matrices are passed as row-major buffers (uint8 for bool, float),
 *   - TBB is replaced by the lock-based shim in cpp/tbbshim, which supports
 *     multi-threaded execution (worker_limit >= 1; 0 = one worker per core),
 *   - results are returned through a plain struct with malloc'd strings.
 *
 * Note: unlike CORELS, the engine unconditionally includes <gmp.h>, so this
 * engine always links against the bundled mini-gmp + mpn shim regardless of
 * SHINRIN_CORELS_NO_GMP (that toggle only affects CORELS' bit-vector type).
 */

#include <cstring>
#include <exception>
#include <new>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include "libspot/include/configuration.hpp"
#include "libspot/include/dataset.hpp"
#include "libspot/include/gosdt.hpp"
#include "libspot/include/matrix.hpp"

extern "C" {

struct ShinrinSpotResult {
    char *model;
    size_t graph_size;
    size_t n_iterations;
    double lower_bound;
    double upper_bound;
    double model_loss;
    double time_elapsed;
    /* Mirrors gosdt::Status declaration order:
     * 0=CONVERGED 1=TIMEOUT 2=NON_CONVERGENCE 3=FALSE_CONVERGENCE
     * 4=UNINITIALIZED */
    int status;
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
 * Fit an optimal sparse decision tree.
 *
 * xy is a rows x cols row-major binary matrix holding [X | binarized y];
 * costs is an n_classes x n_classes float matrix; the feature map maps each
 * original feature to its set of binarized feature indices (flattened into
 * map_sizes/map_indices over n_original_features entries); reference is an
 * optional rows x reference_cols binary matrix of reference predictions.
 *
 * Returns 0 on success (*out populated), 1 on a caught C++ exception
 * (*error_message populated, caller must free), -1 on allocation failure.
 */
int shinrin_spot_fit(
    float regularization, float upperbound_guess, unsigned int time_limit,
    unsigned int model_limit, unsigned int worker_limit, int verbose, int diagnostics, unsigned char depth_budget,
    int reference_lb, int look_ahead, int similar_support, int cancellation,
    int feature_transform, int rule_list, int non_binary,
    const char *trace_path, const char *tree_path, const char *profile_path,
    const unsigned char *xy, size_t rows, size_t cols,
    const float *costs, size_t n_classes,
    size_t n_original_features, const size_t *map_sizes, const size_t *map_indices,
    const unsigned char *reference, size_t reference_cols,
    ShinrinSpotResult *out, char **error_message) {
    try {
        Configuration config;
        config.regularization = regularization;
        if (upperbound_guess > 0.0f) {
            config.upperbound_guess = upperbound_guess;
        }
        config.time_limit = time_limit;
        // worker_limit == 0 means "one worker per available core" (matching
        // upstream's Configuration semantics). The engine's own loop spawns
        // exactly config.worker_limit threads, so resolve it here.
        config.worker_limit =
            worker_limit == 0
                ? (std::thread::hardware_concurrency() > 0 ? std::thread::hardware_concurrency() : 1u)
                : worker_limit;
        config.model_limit = model_limit;
        config.verbose = verbose != 0;
        config.diagnostics = diagnostics != 0;
        config.depth_budget = depth_budget;
        config.reference_LB = reference_lb != 0;
        config.look_ahead = look_ahead != 0;
        config.similar_support = similar_support != 0;
        config.cancellation = cancellation != 0;
        config.feature_transform = feature_transform != 0;
        config.rule_list = rule_list != 0;
        config.non_binary = non_binary != 0;
        config.trace = trace_path != NULL ? trace_path : "";
        config.tree = tree_path != NULL ? tree_path : "";
        config.profile = profile_path != NULL ? profile_path : "";

        Matrix<bool> input(rows, cols);
        for (size_t r = 0; r < rows; r++) {
            for (size_t c = 0; c < cols; c++) {
                input(r, c) = xy[r * cols + c] != 0;
            }
        }

        Matrix<float> cost_matrix(n_classes, n_classes);
        for (size_t i = 0; i < n_classes * n_classes; i++) {
            cost_matrix(i / n_classes, i % n_classes) = costs[i];
        }

        std::vector<std::set<size_t>> feature_map;
        feature_map.reserve(n_original_features);
        size_t offset = 0;
        for (size_t f = 0; f < n_original_features; f++) {
            std::set<size_t> original;
            for (size_t k = 0; k < map_sizes[f]; k++) {
                original.insert(map_indices[offset + k]);
            }
            offset += map_sizes[f];
            feature_map.push_back(std::move(original));
        }

        if (reference != NULL) {
            Matrix<bool> reference_matrix(rows, reference_cols);
            for (size_t r = 0; r < rows; r++) {
                for (size_t c = 0; c < reference_cols; c++) {
                    reference_matrix(r, c) = reference[r * reference_cols + c] != 0;
                }
            }
            Dataset dataset(config, input, cost_matrix, feature_map, reference_matrix);
            gosdt::Result result = gosdt::fit(dataset);
            out->model = dup_c_string(result.model.c_str());
            out->graph_size = result.graph_size;
            out->n_iterations = result.n_iterations;
            out->lower_bound = result.lower_bound;
            out->upper_bound = result.upper_bound;
            out->model_loss = result.model_loss;
            out->time_elapsed = result.time_elapsed;
            out->status = static_cast<int>(result.status);
        } else {
            Dataset dataset(config, input, cost_matrix, feature_map);
            gosdt::Result result = gosdt::fit(dataset);
            out->model = dup_c_string(result.model.c_str());
            out->graph_size = result.graph_size;
            out->n_iterations = result.n_iterations;
            out->lower_bound = result.lower_bound;
            out->upper_bound = result.upper_bound;
            out->model_loss = result.model_loss;
            out->time_elapsed = result.time_elapsed;
            out->status = static_cast<int>(result.status);
        }

        return out->model != NULL ? 0 : -1;
    } catch (std::exception const &e) {
        *error_message = dup_c_string(e.what());
        return *error_message != NULL ? 1 : -1;
    } catch (...) {
        *error_message = dup_c_string("unknown SPOT exception");
        return *error_message != NULL ? 1 : -1;
    }
}

void shinrin_spot_free(void *p) { free(p); }

}  // extern "C"
