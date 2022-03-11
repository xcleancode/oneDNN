/*******************************************************************************
* Copyright 2019-2022 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include <assert.h>
#include <float.h>
#include <math.h>

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/math_utils.hpp"
#include "common/type_helpers.hpp"

#include "cpu/cpu_primitive.hpp"
#include "cpu/ref_io_helper.hpp"

#include "cpu/matmul/sparse/ref_matmul.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace matmul {

status_t ref_sparse_matmul_t::execute(const exec_ctx_t &ctx) const {
    status_t status = status::success;
    auto dst = CTX_OUT_CLEAN_MEM(float *, DNNL_ARG_DST, status);
    CHECK(status);

    const auto src_d = ctx.memory_mdw(DNNL_ARG_SRC, pd()->src_md());
    const auto weights_d = ctx.memory_mdw(DNNL_ARG_WEIGHTS, pd()->weights_md());
    const auto dst_d = ctx.memory_mdw(DNNL_ARG_DST, pd()->dst_md());
    // TODO: add bias support
    //    const auto bia_d = ctx.memory_mdw(DNNL_ARG_BIAS, pd()->weights_md(1));

    //const bool non_default_attrs = !pd()->attr()->has_default_values();

    //    printf("values:\n");
    //    for (size_t i = 0; i < weights_d.size(0) / sizeof(float); i++)
    //        printf("%f ", wei_values[i]);
    //
    //    printf("\nindices:\n");
    //    for (size_t i = 0; i < weights_d.size(1); i++)
    //        printf("%d ", wei_indices[i]);
    //
    //    printf("\npointer:\n");
    //    for (size_t i = 0; i < weights_d.size(2); i++)
    //        printf("%d ", wei_pointers[i]);
    //    printf("\n");

    const dim_t M = dst_d.dims()[0];
    const dim_t N = dst_d.dims()[1];
    const dim_t K = src_d.dims()[1];

    // Kernel for sparse mm
    // dst(m,n) = src(m,k) * wei(k,n)

    // mm kernel

    parallel_nd(M, N, [&](dim_t i, dim_t j) { dst[i * N + j] = 0.0f; });

    if (weights_d.is_sparse_desc()) {
        const auto src = CTX_IN_MEM(const float *, DNNL_ARG_SRC);
        const auto wei_values
                = CTX_IN_SPARSE_MEM(const float *, DNNL_ARG_WEIGHTS, 0);
        // TODO: generalize data types.
        const auto wei_indices
                = CTX_IN_SPARSE_MEM(const int32_t *, DNNL_ARG_WEIGHTS, 1);
        const auto wei_pointers
                = CTX_IN_SPARSE_MEM(const int32_t *, DNNL_ARG_WEIGHTS, 2);

        parallel_nd(M, [&](dim_t m) {
            for (int32_t k = 0; k < K; k++) {
                for (int32_t nB = wei_pointers[k]; nB < wei_pointers[k + 1];
                        nB++) {
                    const int32_t kA = m * K + k;
                    int32_t n = wei_indices[nB];
                    int32_t nC = m * N + n;
                    dst[nC] = dst[nC] + src[kA] * wei_values[nB];
                }
            }
        });
    } else if (src_d.is_sparse_desc()) {
        const auto weights = CTX_IN_MEM(const float *, DNNL_ARG_WEIGHTS);
        const auto src_values
                = CTX_IN_SPARSE_MEM(const float *, DNNL_ARG_SRC, 0);
        // TODO: generalize data types.
        const auto src_indices
                = CTX_IN_SPARSE_MEM(const int32_t *, DNNL_ARG_SRC, 1);
        const auto src_pointers
                = CTX_IN_SPARSE_MEM(const int32_t *, DNNL_ARG_SRC, 2);

        parallel_nd(M, [&](dim_t m) {
            for (dim_t kA = src_pointers[m]; kA < src_pointers[m + 1]; kA++) {
                dim_t k = src_indices[kA];
                for (dim_t n = 0; n < N; n++) {
                    dim_t nC = m * N + n;
                    dim_t nB = k * N + n;
                    dst[nC] = dst[nC] + src_values[kA] * weights[nB];
                }
            }
        });
    }

    return status::success;
}

} // namespace matmul
} // namespace cpu
} // namespace impl
} // namespace dnnl
