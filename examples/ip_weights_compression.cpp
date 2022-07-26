/*******************************************************************************
* Copyright 2022 Intel Corporation
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

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include "dnnl.hpp"
#include "example_utils.hpp"

using namespace dnnl;

using tag = memory::format_tag;
using dt = memory::data_type;

void ip_weights_compression(dnnl::engine::kind engine_kind) {
    dnnl::engine engine(engine_kind, 0);
    dnnl::stream engine_stream(engine);

    const memory::dim N = 3, IC = 3, OC = 96;

    memory::dims src_dims = {N, IC};
    memory::dims weights_dims = {OC, IC};
    memory::dims dst_dims = {N, OC};

    std::vector<float> src_data(product(src_dims));
    std::vector<float> weights_data(product(weights_dims));
    std::vector<float> dst_data(product(dst_dims));

    std::generate(src_data.begin(), src_data.end(), []() {
        static int i = 0;
        return std::cos(i++ / 10.f);
    });
    std::generate(weights_data.begin(), weights_data.end(), []() {
        static int i = 0;
        return i++ % 2 == 0 ? 0 : std::sin(i++ * 2.f);
    });

    const memory::dim nnz = std::count_if(weights_data.begin(),
            weights_data.end(), [](float v) { return v != 0.0f; });

    auto src_md = memory::desc(src_dims, dt::f32, tag::nc);
    auto dst_md = memory::desc(dst_dims, dt::f32, tag::nc);

    auto src_mem = memory(src_md, engine);
    auto dst_mem = memory(dst_md, engine);

    auto user_src_mem = memory({src_dims, dt::f32, tag::nc}, engine);
    auto user_weights_mem = memory({weights_dims, dt::f32, tag::oi}, engine);
    auto user_dst_mem = memory({dst_dims, dt::f32, tag::nc}, engine);

    write_to_dnnl_memory(src_data.data(), src_mem);
    write_to_dnnl_memory(weights_data.data(), user_weights_mem);

    auto inner_product_src_md = memory::desc(src_dims, dt::u8, tag::any);
    // Explicitly request a packed sparse format (similar to tag "any").
    auto inner_product_weights_md
            = memory::desc(weights_dims, dt::s8, memory::desc::packed(nnz));
    auto inner_product_dst_md = memory::desc(dst_dims, dt::u8, tag::any);

    auto inner_product_d = inner_product_forward::desc(
            prop_kind::forward_inference, inner_product_src_md,
            inner_product_weights_md, inner_product_dst_md);

    auto inner_product_pd
            = inner_product_forward::primitive_desc(inner_product_d, engine);

    auto inner_product_src_mem = user_src_mem;
    auto inner_product_weights_mem = user_weights_mem;
    auto inner_product_dst_mem = user_dst_mem;

    if (inner_product_pd.src_desc() != user_src_mem.get_desc()) {
        inner_product_src_mem = memory(inner_product_pd.src_desc(), engine);
        reorder(user_src_mem, inner_product_src_mem)
                .execute(engine_stream, user_src_mem, inner_product_src_mem);
    }

    // Compress weights.
    if (inner_product_pd.weights_desc() != user_weights_mem.get_desc()) {
        inner_product_weights_mem
                = memory(inner_product_pd.weights_desc(), engine);
        reorder(user_weights_mem, inner_product_weights_mem)
                .execute(engine_stream, user_weights_mem,
                        inner_product_weights_mem);
    }

    if (inner_product_pd.dst_desc() != user_dst_mem.get_desc()) {
        inner_product_dst_mem = memory(inner_product_pd.dst_desc(), engine);
        reorder(user_dst_mem, inner_product_dst_mem)
                .execute(engine_stream, user_dst_mem, inner_product_dst_mem);
    }

    auto inner_product_prim = inner_product_forward(inner_product_pd);

    std::unordered_map<int, memory> inner_product_args;
    inner_product_args.insert({DNNL_ARG_SRC, inner_product_src_mem});
    inner_product_args.insert({DNNL_ARG_WEIGHTS, inner_product_weights_mem});
    inner_product_args.insert({DNNL_ARG_DST, inner_product_dst_mem});

    inner_product_prim.execute(engine_stream, inner_product_args);
    engine_stream.wait();

    read_from_dnnl_memory(dst_data.data(), dst_mem);
}

int main(int argc, char **argv) {
    return handle_example_errors(
            ip_weights_compression, parse_engine_kind(argc, argv));
}
