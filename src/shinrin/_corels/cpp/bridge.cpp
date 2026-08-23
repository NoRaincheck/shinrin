/*
 * C ABI bridge between shinrin's native extension (PyO3/Rust) and the
 * vendored CORELS C++ sources. This replaces upstream pycorels' Cython
 * binding (corels/_corels.pyx) while keeping identical semantics,
 * including the module-global fit state and its lifecycle.
 *
 * Compiled with -DGMP, where `<gmp.h>` resolves through `gmpshim/gmp.h` to
 * the vendored mini-gmp (see `minigmp/README.md`): the GMP code path is
 * active with no dependency on a system libgmp.
 *
 * Provenance: derived from pycorels (GPL-3.0), https://github.com/corels/pycorels
 */

#include <set>
#include <string>
#include <vector>

#include <stdlib.h>
#include <string.h>

#include "corels/rule.h"
#include "corels/run.h"
#include "corels/pmap.h"
#include "corels/cache.h"
#include "corels/queue.h"
#include "mining/utils.hh"

namespace {

/*
 * Fit state, mirroring the globals in _corels.pyx. Access is serialized by
 * the Python GIL on the caller side, exactly as with upstream.
 */
rule_t* g_rules = NULL;
rule_t* g_labels_vecs = NULL;
rule_t* g_minor = NULL;
int g_nrules = 0;
PermutationMap* g_pmap = NULL;
CacheTree* g_tree = NULL;
Queue* g_queue = NULL;
double g_init = 0.0;
std::set<std::string> g_verbosity;

/* Return codes understood by the Rust binding layer. */
const int SHINRIN_CORELS_OK = 0;
const int SHINRIN_CORELS_RUN_FAILED = 1;
const int SHINRIN_CORELS_MEMORY = -1;
const int SHINRIN_CORELS_VALUE = -2;

void free_vector(rule_t* vs, int count) {
    if (vs == NULL) {
        return;
    }
    for (int i = 0; i < count; i++) {
        rule_vfree(&vs[i].truthtable);
        if (vs[i].ids) {
            free(vs[i].ids);
        }
        if (vs[i].features) {
            free(vs[i].features);
        }
    }
    free(vs);
}

/* Convert a row-major binary matrix into rule_t bit vectors, via the same
 * ascii_to_vector round-trip used by upstream (_to_vector in _corels.pyx),
 * so bit-order semantics are preserved exactly. Returns NULL on failure. */
rule_t* to_vectors(const unsigned char* X, int d0, int d1, int* ncount_out) {
    rule_t* vectors = (rule_t*)malloc(d0 * sizeof(rule_t));
    if (vectors == NULL) {
        return NULL;
    }

    for (int i = 0; i < d0; i++) {
        char* arrstr = (char*)malloc(d1 + 1);
        if (arrstr == NULL) {
            free_vector(vectors, i);
            return NULL;
        }
        for (int j = 0; j < d1; j++) {
            arrstr[j] = X[(size_t)i * d1 + j] ? '1' : '0';
        }
        arrstr[d1] = '\0';

        int ncount = d1;
        int nones = 0;
        if (ascii_to_vector(arrstr, ncount, &ncount, &nones, &vectors[i].truthtable) != 0) {
            free(arrstr);
            free_vector(vectors, i);
            return NULL;
        }
        free(arrstr);

        *ncount_out = ncount;

        vectors[i].ids = NULL;
        vectors[i].features = NULL;
        vectors[i].cardinality = 1;
        vectors[i].support = nones;
    }

    return vectors;
}

char* dup_string(const char* s) {
    if (s == NULL) {
        return NULL;
    }
    size_t n = strlen(s);
    char* out = (char*)malloc(n + 1);
    if (out != NULL) {
        memcpy(out, s, n + 1);
    }
    return out;
}

} // namespace

extern "C" {

/*
 * Mine rules, compute the minority bound, and initialize the CORELS search.
 * Mirrors fit_wrap_begin. samples is [nsamples x nfeatures], labels is
 * [2 x nsamples] (row 0 = inverted label, row 1 = label).
 *
 * Returns SHINRIN_CORELS_OK on success, SHINRIN_CORELS_RUN_FAILED when
 * run_corels_begin reports failure, SHINRIN_CORELS_MEMORY on allocation
 * failure, and SHINRIN_CORELS_VALUE on data inconsistencies.
 */
int shinrin_corels_begin(const unsigned char* samples, int nsamples, int nfeatures,
                         const unsigned char* labels, const char** features,
                         int nfeatures_names, int max_card, double min_support,
                         const char* verbosity_str, int mine_verbose, int minor_verbose,
                         double c, int policy, int map_type, int ablation,
                         int calculate_size) {
    int ncount = 0;
    rule_t* samples_vecs = to_vectors(samples, nsamples, nfeatures, &ncount);

    if (samples_vecs == NULL) {
        return SHINRIN_CORELS_MEMORY;
    }

    if (ncount > nfeatures_names) {
        free_vector(samples_vecs, nsamples);
        return SHINRIN_CORELS_VALUE;
    }

    char** features_vec = (char**)malloc(ncount * sizeof(char*));
    if (features_vec == NULL) {
        free_vector(samples_vecs, nsamples);
        return SHINRIN_CORELS_MEMORY;
    }

    for (int i = 0; i < ncount; i++) {
        features_vec[i] = dup_string(features[i]);
        if (features_vec[i] == NULL) {
            for (int j = 0; j < i; j++) {
                free(features_vec[j]);
            }
            free(features_vec);
            free_vector(samples_vecs, nsamples);
            return SHINRIN_CORELS_MEMORY;
        }
    }

    if (g_rules != NULL) {
        free_vector(g_rules, g_nrules);
        g_rules = NULL;
    }
    g_nrules = 0;

    int r = mine_rules(features_vec, samples_vecs, ncount, nsamples, max_card,
                       min_support, &g_rules, mine_verbose);

    for (int i = 0; i < ncount; i++) {
        free(features_vec[i]);
    }
    free(features_vec);
    free_vector(samples_vecs, nsamples);

    if (r == -1 || g_rules == NULL) {
        return SHINRIN_CORELS_MEMORY;
    }
    g_nrules = r;

    char* verbosity = dup_string(verbosity_str);
    if (verbosity == NULL) {
        free_vector(g_rules, g_nrules);
        g_rules = NULL;
        g_nrules = 0;
        return SHINRIN_CORELS_MEMORY;
    }

    if (g_labels_vecs != NULL) {
        free_vector(g_labels_vecs, 2);
        g_labels_vecs = NULL;
    }

    int nsamples_chk = 0;
    g_labels_vecs = to_vectors(labels, 2, nsamples, &nsamples_chk);
    if (g_labels_vecs == NULL) {
        free(verbosity);
        free_vector(g_rules, g_nrules);
        g_rules = NULL;
        g_nrules = 0;
        return SHINRIN_CORELS_MEMORY;
    }

    if (nsamples_chk != nsamples) {
        free(verbosity);
        free_vector(g_labels_vecs, 2);
        g_labels_vecs = NULL;
        free_vector(g_rules, g_nrules);
        g_rules = NULL;
        g_nrules = 0;
        return SHINRIN_CORELS_VALUE;
    }

    g_labels_vecs[0].features = dup_string("label=0");
    g_labels_vecs[1].features = dup_string("label=1");
    if (g_labels_vecs[0].features == NULL || g_labels_vecs[1].features == NULL) {
        free(verbosity);
        free_vector(g_labels_vecs, 2);
        g_labels_vecs = NULL;
        free_vector(g_rules, g_nrules);
        g_rules = NULL;
        g_nrules = 0;
        return SHINRIN_CORELS_MEMORY;
    }

    if (g_minor != NULL) {
        free_vector(g_minor, 1);
        g_minor = NULL;
    }

    g_minor = (rule_t*)malloc(sizeof(rule_t));
    if (g_minor == NULL) {
        free(verbosity);
        free_vector(g_labels_vecs, 2);
        g_labels_vecs = NULL;
        free_vector(g_rules, g_nrules);
        g_rules = NULL;
        g_nrules = 0;
        return SHINRIN_CORELS_MEMORY;
    }

    int mr = minority(g_rules, g_nrules, g_labels_vecs, nsamples, g_minor, minor_verbose);
    if (mr != 0) {
        free(verbosity);
        free_vector(g_labels_vecs, 2);
        g_labels_vecs = NULL;
        free_vector(g_rules, g_nrules);
        g_rules = NULL;
        g_nrules = 0;
        return SHINRIN_CORELS_MEMORY;
    }

    int rb = run_corels_begin(c, verbosity, policy, map_type, ablation, calculate_size,
                              g_nrules, 2, nsamples, g_rules, g_labels_vecs, g_minor, 0,
                              NULL, g_pmap, g_tree, g_queue, g_init, g_verbosity);
    free(verbosity);

    if (rb == -1) {
        free_vector(g_labels_vecs, 2);
        g_labels_vecs = NULL;
        free_vector(g_minor, 1);
        g_minor = NULL;
        free_vector(g_rules, g_nrules);
        g_rules = NULL;
        g_nrules = 0;
        return SHINRIN_CORELS_RUN_FAILED;
    }

    return SHINRIN_CORELS_OK;
}

/*
 * Run the search loop. Mirrors fit_wrap_loop: returns 1 while the search
 * should continue, 0 once finished (or on internal failure, as upstream).
 */
int shinrin_corels_loop(size_t max_num_nodes) {
    return (run_corels_loop(max_num_nodes, g_pmap, g_tree, g_queue) != -1) ? 1 : 0;
}

/*
 * Extract the optimal rulelist and tear down the search. Mirrors
 * fit_wrap_end.
 *
 * Upstream reads rule metadata (antecedent ids) from the global mined-rules
 * array before freeing it, so this function flattens everything the Python
 * layer needs before tearing down:
 *
 *   - out_lens:   antecedent count per emitted rule (one entry per element
 *                 of the optimal rulelist below the rules-array bound)
 *   - out_ants:   the concatenated antecedent id lists of those rules
 *   - out_classes: the optimal predictions, rulelist length + 1 entries
 *                 (the last one being the default prediction)
 *
 * All three arrays are malloc'd and must be released with
 * shinrin_corels_free_ints.
 */
double shinrin_corels_end(int early, int** out_lens, int* out_lens_len, int** out_ants,
                          int* out_ants_len, int** out_classes, int* out_classes_len) {
    std::vector<int> rulelist;
    std::vector<int> classes;
    double objective = run_corels_end(&rulelist, &classes, early, 0, NULL, NULL, NULL,
                                      g_pmap, g_tree, g_queue, g_init, g_verbosity);

    g_pmap = NULL;
    g_tree = NULL;
    g_queue = NULL;

    /* Flatten antecedents while the rules array is still alive. */
    std::vector<int> lens;
    std::vector<int> ants;
    for (size_t i = 0; i < rulelist.size(); i++) {
        if (rulelist[i] < g_nrules) {
            const rule_t& r = g_rules[rulelist[i]];
            lens.push_back(r.cardinality);
            for (int j = 0; j < r.cardinality; j++) {
                ants.push_back(r.ids[j]);
            }
        }
    }

    *out_lens_len = (int)lens.size();
    *out_ants_len = (int)ants.size();
    *out_classes_len = (int)classes.size();

    size_t lens_bytes = lens.size() * sizeof(int);
    size_t ants_bytes = ants.size() * sizeof(int);
    size_t cl_bytes = classes.size() * sizeof(int);
    *out_lens = (int*)(lens_bytes ? malloc(lens_bytes) : NULL);
    *out_ants = (int*)(ants_bytes ? malloc(ants_bytes) : NULL);
    *out_classes = (int*)(cl_bytes ? malloc(cl_bytes) : NULL);
    if ((lens_bytes && *out_lens == NULL) || (ants_bytes && *out_ants == NULL) ||
        (cl_bytes && *out_classes == NULL)) {
        free(*out_lens);
        free(*out_ants);
        free(*out_classes);
        *out_lens = NULL;
        *out_ants = NULL;
        *out_classes = NULL;
    } else {
        if (lens_bytes) {
            memcpy(*out_lens, lens.data(), lens_bytes);
        }
        if (ants_bytes) {
            memcpy(*out_ants, ants.data(), ants_bytes);
        }
        if (cl_bytes) {
            memcpy(*out_classes, classes.data(), cl_bytes);
        }
    }

    /* Exiting early skips cleanup, as upstream. */
    if (early == 0) {
        if (g_labels_vecs != NULL) {
            free_vector(g_labels_vecs, 2);
        }
        if (g_minor != NULL) {
            free_vector(g_minor, 1);
        }
        if (g_rules != NULL) {
            free_vector(g_rules, g_nrules);
        }
        g_labels_vecs = NULL;
        g_minor = NULL;
        g_rules = NULL;
        g_nrules = 0;
    }

    return objective;
}

/* Free arrays returned by shinrin_corels_end. */
void shinrin_corels_free_ints(void* p) {
    free(p);
}

/* 1 when the engine was compiled with the GMP (mini-gmp) bit vectors,
 * 0 when using the word-array fallback. */
int shinrin_corels_gmp_enabled(void) {
#ifdef GMP
    return 1;
#else
    return 0;
#endif
}

} // extern "C"
