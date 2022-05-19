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

#include <assert.h>

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/math_utils.hpp"
#include "common/nstl.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/x64/jit_generator.hpp"

#include "cpu/x64/matmul/sparse/jit_uni_matmul.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {
namespace matmul {

using namespace dnnl::impl::data_type;
using namespace Xbyak;

struct sparse_matmul_kernel_t : public jit_generator {
    struct call_params_t {
        size_t n_size_in_bytes;
        float src_val;
        const float *wei, *dst;
        size_t need_zero_dst;
        size_t only_nullify;
    };

    sparse_matmul_kernel_t(size_t vlen, const matmul_pd_t *pd) : vlen_(vlen) {
        simd_w_ = vlen_ / data_type_size();
        tail_size_ = pd->dst_md()->dims[1] % simd_w();
    }

    virtual ~sparse_matmul_kernel_t() = default;

    void operator()(const call_params_t *p) {
        return jit_generator::operator()(p);
    }

    size_t simd_w() const { return simd_w_; }
    size_t vlen() const { return vlen_; }
    size_t tail_size() const { return tail_size_; }

    size_t data_type_size() const { return sizeof(float); }

protected:
    size_t vlen_;
    size_t simd_w_;
    size_t tail_size_;
};

template <cpu_isa_t isa>
struct jit_uni_sparse_matmul_kernel_t : public sparse_matmul_kernel_t {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_sparse_matmul_kernel_t)

    using sparse_matmul_kernel_t::data_type_size;
    using sparse_matmul_kernel_t::simd_w;
    using sparse_matmul_kernel_t::tail_size;
    using sparse_matmul_kernel_t::vlen;

    using Vmm = typename cpu_isa_traits<isa>::Vmm;
    const AddressFrame &vmmword = (isa == avx2) ? yword : zword;

    Reg64 reg_param = abi_param1;

    Reg64 reg_only_nullify = rbx;
    Reg64 reg_need_zero_dst = r8;

    Reg64 reg_wei = r9;
    Reg64 reg_dst = r10;
    Reg64 reg_offset_n = r11;
    Reg64 reg_n_size_in_bytes = r12;
    Reg64 reg_reverse_n_size_in_bytes = r13;
    Reg64 reg_tmp = r14;
    Reg64 reg_elt_inj_table = r15;

    size_t unroll_regs_ = isa == avx512_core ? 4 : 4;

    Opmask tail_opmask = Opmask(2);
    Vmm tail_vmask = Vmm(0);

    Vmm vreg_src_val = Vmm(isa == avx512_core ? 19 : 11);
    Xmm xreg_src_val = Xmm(11);
    Vmm vreg_zero = Vmm(isa == avx512_core ? 20 : 12);

    void load_kernel_params() {
#define PARAM_OFF(x) offsetof(call_params_t, x)
        mov(reg_n_size_in_bytes, ptr[reg_param + PARAM_OFF(n_size_in_bytes)]);
        mov(reg_tmp, ptr[reg_param + PARAM_OFF(src_val)]);
        uni_vmovq(xreg_src_val, reg_tmp);
        uni_vbroadcastss(vreg_src_val, xreg_src_val);

        mov(reg_wei, ptr[reg_param + PARAM_OFF(wei)]);
        mov(reg_dst, ptr[reg_param + PARAM_OFF(dst)]);
        mov(reg_need_zero_dst, ptr[reg_param + PARAM_OFF(need_zero_dst)]);
        mov(reg_only_nullify, ptr[reg_param + PARAM_OFF(only_nullify)]);
#undef PARAM_OFF
    }

    Address wei_ptr(size_t offt = 0) {
        return vmmword[reg_wei + reg_offset_n + offt];
    }

    Address dst_ptr(size_t offt = 0) {
        return vmmword[reg_dst + reg_offset_n + offt];
    }

    void load_tail(const Zmm &dst, const Address &src) {
        uni_vmovups_tail(dst, tail_opmask, src);
    }

    void load_tail(const Ymm &dst, const Address &src) {
        uni_vmovups_tail(dst, tail_vmask, src);
    }

    void store_tail(const Address &dst, const Zmm &src) {
        uni_vmovups_tail(dst, tail_opmask, src);
    }

    void store_tail(const Address &dst, const Ymm &src) {
        uni_vmovups_tail(dst, tail_vmask, src);
    }

    void nullify_dst(int i, bool tail) {
        const int offt = simd_w() * i * data_type_size();
        if (tail) {
            store_tail(dst_ptr(offt), vreg_zero);
        } else {
            uni_vmovups(dst_ptr(offt), vreg_zero);
        }
    }

    void compute(int unroll, bool tail = false) {
        for (int i = 0; i < unroll; i++) {
            Label zero_dst, compute_dst;
            Label only_nullify, end;

            cmp(reg_only_nullify, 1);
            je(only_nullify, T_NEAR);

            Vmm vreg_tmp_wei = Vmm(2 * i + 1);
            Vmm vreg_tmp_dst = Vmm(2 * i + 2);
            int offt = simd_w() * i * data_type_size();
            if (tail) {
                load_tail(vreg_tmp_wei, wei_ptr(offt));
                load_tail(vreg_tmp_dst, dst_ptr(offt));
            } else {
                uni_vmovups(vreg_tmp_wei, wei_ptr(offt));
                uni_vmovups(vreg_tmp_dst, dst_ptr(offt));
            }


            cmp(reg_need_zero_dst, 1);
            je(zero_dst, T_NEAR);
            jmp(compute_dst, T_NEAR);

            L(zero_dst);
            uni_vpxor(vreg_tmp_dst, vreg_tmp_dst, vreg_tmp_dst);

            L(compute_dst);
            uni_vfmadd231ps(vreg_tmp_dst, vreg_src_val, vreg_tmp_wei);

            if (tail) {
                store_tail(dst_ptr(offt), vreg_tmp_dst);
            } else {
                uni_vmovups(dst_ptr(offt), vreg_tmp_dst);
            }
            jmp(end);

            L(only_nullify);
            nullify_dst(i, tail);
            L(end);
        }
    }

    void prepare_tail_mask_avx2() {
        if (tail_size() == 0) return;

        static const uint32_t mask_f32[14]
                = {0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff,
                        0xffffffff, 0xffffffff, 0, 0, 0, 0, 0, 0, 0};

        mov(reg_tmp, reinterpret_cast<size_t>(&mask_f32[7 - tail_size()]));
        vmovups(tail_vmask, ptr[reg_tmp]);
    }

    void prepare_tail_mask_avx512() {
        if (tail_size() == 0) return;

        const int mask_f32 = (1 << tail_size()) - 1;

        Reg32 regw_tmp = reg_tmp.cvt32();
        mov(regw_tmp, mask_f32);
        kmovd(tail_opmask, regw_tmp);
    }

    void loop_over_n() {
        uni_vpxor(vreg_zero, vreg_zero, vreg_zero);
        Label unroll_loop, unroll_loop_tail, nelems_tail, end;

        mov(reg_reverse_n_size_in_bytes, reg_n_size_in_bytes);
        xor_(reg_offset_n, reg_offset_n);
        size_t vec_size = simd_w() * data_type_size();
        L(unroll_loop);
        {
            size_t offt = unroll_regs_ * vec_size;
            cmp(reg_reverse_n_size_in_bytes, offt);
            jl(unroll_loop_tail, T_NEAR);

            compute(unroll_regs_);
            sub(reg_reverse_n_size_in_bytes, offt);
            add(reg_offset_n, offt);
            jmp(unroll_loop);
        }

        L(unroll_loop_tail);
        {
            cmp(reg_reverse_n_size_in_bytes, vec_size);
            jl(nelems_tail, T_NEAR);

            compute(1);
            sub(reg_reverse_n_size_in_bytes, vec_size);
            add(reg_offset_n, vec_size);
            jmp(unroll_loop_tail);
        }

        L(nelems_tail);
        {
            cmp(reg_reverse_n_size_in_bytes, 1);
            jl(end, T_NEAR);
            compute(1, true);
        }

        L(end);
    }

    void generate() override {
        preamble();
        if (isa == avx512_core)
            prepare_tail_mask_avx512();
        else
            prepare_tail_mask_avx2();
        load_kernel_params();
        loop_over_n();
        postamble();
    }

    jit_uni_sparse_matmul_kernel_t(const matmul_pd_t *pd)
        : sparse_matmul_kernel_t(cpu_isa_traits<isa>::vlen, pd) {}
    virtual ~jit_uni_sparse_matmul_kernel_t() = default;
};

status_t jit_uni_sparse_matmul_t::init(engine_t *engine) {
    if (mayiuse(avx512_core)) {
        using kernel_t = jit_uni_sparse_matmul_kernel_t<avx512_core>;
        kernel_ = std::unique_ptr<kernel_t> {new kernel_t(pd())};
    } else if (mayiuse(avx2)) {
        using kernel_t = jit_uni_sparse_matmul_kernel_t<avx2>;
        kernel_ = std::unique_ptr<kernel_t> {new kernel_t(pd())};
    }
    if (!kernel_) return status::runtime_error;

    CHECK(kernel_->create_kernel());
    return status::success;
}

jit_uni_sparse_matmul_t::jit_uni_sparse_matmul_t(const pd_t *apd)
    : primitive_t(apd) {}
jit_uni_sparse_matmul_t::~jit_uni_sparse_matmul_t() = default;

status_t jit_uni_sparse_matmul_t::execute(const exec_ctx_t &ctx) const {
    const auto *weights = CTX_IN_MEM(const float *, DNNL_ARG_WEIGHTS);
    const auto *src_values = CTX_IN_SPARSE_MEM(const float *, DNNL_ARG_SRC, 0);
    const auto *src_indices
            = CTX_OUT_SPARSE_MEM(const int32_t *, DNNL_ARG_SRC, 1);
    const auto *src_pointers
            = CTX_OUT_SPARSE_MEM(const int32_t *, DNNL_ARG_SRC, 2);

    status_t status = status::success;
    auto dst = CTX_OUT_CLEAN_MEM(float *, DNNL_ARG_DST, status);
    CHECK(status);

    const memory_desc_wrapper src_d(pd()->src_md());
    const memory_desc_wrapper dst_d(pd()->dst_md());

    const dim_t M = dst_d.dims()[0];
    const dim_t N = dst_d.dims()[1];

    const auto data_type_size = kernel_->data_type_size();

    parallel_nd(M, [&](dim_t m) {
        const bool row_has_nnz = src_pointers[m] < src_pointers[m + 1];
        if (!row_has_nnz) {
            // Need to nullify the output row.
            sparse_matmul_kernel_t::call_params_t p;
            p.n_size_in_bytes = N * data_type_size;
            p.src_val = 0.0f;
            p.wei = nullptr;
            p.dst = dst + (m * N);
            p.need_zero_dst = false;
            p.only_nullify = true;
            (*kernel_)(&p);
        }

        for (dim_t kA = src_pointers[m]; kA < src_pointers[m + 1]; kA++) {
            const dim_t k = src_indices[kA];
            sparse_matmul_kernel_t::call_params_t p;
            p.n_size_in_bytes = N * data_type_size;
            p.src_val = src_values[kA];
            p.wei = weights + (k * N);
            p.dst = dst + (m * N);
            p.need_zero_dst = (kA == src_pointers[m]);
            p.only_nullify = 0;
            (*kernel_)(&p);
        }
    });

    return status::success;
}

} // namespace matmul
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
