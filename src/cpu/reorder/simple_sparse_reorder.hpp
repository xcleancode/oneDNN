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

#ifndef CPU_REORDER_SIMPLE_SPARSE_REORDER_HPP
#define CPU_REORDER_SIMPLE_SPARSE_REORDER_HPP

#include <assert.h>

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/math_utils.hpp"
#include "common/primitive.hpp"
#include "common/primitive_attr.hpp"
#include "common/tag_traits.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/cpu_primitive.hpp"
#include "cpu/reorder/cpu_reorder_pd.hpp"

#include "cpu/simple_q10n.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

// The following cases can be covered:
//
// - sparse_tag -> sparse_tag
// - encoding -> encoding
//
// - sparse_tag -> dense_tag
// - dense_tag -> sparse_tag
//
// - sparse_tag -> encoding
// - encoding -> sparse_tag
//
// - dense_tag -> encoding
// - encoding -> dense_tag
#define SIMPLE_SPARSE_REORDER_TEMPL_DECL \
    impl::data_type_t type_i, typename fmt_i_t, fmt_i_t fmt_i, \
            impl::data_type_t type_o, typename fmt_o_t, fmt_o_t fmt_o, \
            bool order_keep

#define SIMPLE_SPARSE_REORDER_TEMPL_CALL \
    type_i, fmt_i_t, fmt_i, type_o, fmt_o_t, fmt_o, order_keep

// TODO: move common code to reorder_utils.hpp.
namespace sparse_spec {
struct reference {};
} // namespace sparse_spec

namespace sparse_inputs_order {
constexpr bool keep = true;
constexpr bool reverse = false;
constexpr bool any = keep;
} // namespace sparse_inputs_order

template <SIMPLE_SPARSE_REORDER_TEMPL_DECL, typename spec = void>
struct simple_sparse_reorder_impl {};

namespace {
template <typename T>
constexpr bool is_format_tag(T) {
    return std::is_same<T, format_tag_t>::value ? true : false;
}
} // namespace

using namespace data_type;

template <SIMPLE_SPARSE_REORDER_TEMPL_DECL>
struct simple_sparse_reorder_impl<SIMPLE_SPARSE_REORDER_TEMPL_CALL,
        typename utils::enable_if<is_format_tag(fmt_i)
                        && fmt_i == format_tag::ab && !is_format_tag(fmt_o)
                        && fmt_o == sparse_encoding::csr,
                sparse_spec::reference>::type> {

    static bool is_applicable(const memory_desc_wrapper &input_d,
            const memory_desc_wrapper &output_d, const primitive_attr_t *attr) {
        using namespace data_type;
        using namespace utils;
        if (input_d.has_runtime_dims_or_strides()) return false;

        return input_d.matches_tag(format_tag::ab) && output_d.is_sparse_desc()
                && output_d.sparse_desc().encoding == sparse_encoding::csr
                && input_d.data_type() == f32 && output_d.data_type() == f32;

        return true;
    }

    static status_t execute(
            const cpu_reorder_pd_t *pd_object, const exec_ctx_t &ctx) {
        auto pd = [pd_object]() { return pd_object; };

        auto input = CTX_IN_MEM(const data_t<type_i> *, DNNL_ARG_FROM);
        auto output_values
                = CTX_OUT_SPARSE_MEM(data_t<type_o> *, DNNL_ARG_TO, 0);
        auto output_indices = CTX_OUT_SPARSE_MEM(data_t<s8> *, DNNL_ARG_TO, 1);
        auto output_pointers = CTX_OUT_SPARSE_MEM(data_t<s8> *, DNNL_ARG_TO, 2);

        const auto input_d = ctx.memory_mdw(DNNL_ARG_FROM, pd()->src_md());

        const size_t M = input_d.dims()[0];
        const size_t N = input_d.dims()[1];
        const size_t LD = N;

        size_t nnz_cnt = 0;
        output_pointers[0] = 0;
        int8_t *op_ptr = &output_pointers[0];

        for (size_t i = 0; i < M; i++) {
            size_t nnz_per_row = 0;
            for (size_t j = 0; j < N; j++) {
                const auto val = input[i * LD + j];
                if (val != 0.0f) {
                    output_values[nnz_cnt] = val;
                    output_indices[nnz_cnt] = j;
                    nnz_cnt++;
                    nnz_per_row++;
                }
            }
            const int8_t curr_ptr = *op_ptr;
            op_ptr++;
            *op_ptr = curr_ptr + nnz_per_row;
        }
        return status::success;
    }
};

/* high level class declaration */
template <SIMPLE_SPARSE_REORDER_TEMPL_DECL, typename spec = void>
struct simple_sparse_reorder_t : public primitive_t {
    struct pd_t : public cpu_reorder_pd_t {
        using cpu_reorder_pd_t::cpu_reorder_pd_t;

        DECLARE_COMMON_PD_T("simple_sparse:any", simple_sparse_reorder_t);

    private:
        static status_t create(reorder_pd_t **reorder_pd, engine_t *engine,
                const primitive_attr_t *attr, engine_t *src_engine,
                const memory_desc_t *src_md, engine_t *dst_engine,
                const memory_desc_t *dst_md) {
            const bool args_ok = true && src_md->data_type == type_i
                    && dst_md->data_type == type_o && attr->has_default_values()
                    && simple_sparse_reorder_impl<
                            SIMPLE_SPARSE_REORDER_TEMPL_CALL,
                            spec>::is_applicable(src_md, dst_md, attr);
            if (!args_ok) return status::invalid_arguments;

            auto _pd = new pd_t(attr, src_engine->kind(), src_md,
                    dst_engine->kind(), dst_md);
            if (_pd == nullptr) return status::out_of_memory;
            if (_pd->init(engine, src_engine, dst_engine) != status::success) {
                delete _pd;
                return status::unimplemented;
            }

            _pd->init_scratchpad_md();
            return safe_ptr_assign(*reorder_pd, _pd);
        }
        friend dnnl::impl::impl_list_item_t;
    };

    simple_sparse_reorder_t(const pd_t *apd) : primitive_t(apd) {}

    status_t execute(const exec_ctx_t &ctx) const override {
        return simple_sparse_reorder_impl<SIMPLE_SPARSE_REORDER_TEMPL_CALL,
                spec>::execute(pd(), ctx);
    }

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
};

#undef SIMPLE_SPARSE_REORDER_TEMPL_DECL
#undef SIMPLE_SPARSE_REORDER_TEMPL_CALL

} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
