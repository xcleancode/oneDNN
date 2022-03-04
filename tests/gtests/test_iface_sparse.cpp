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

#include "dnnl_test_common.hpp"
#include "gtest/gtest.h"

#include "oneapi/dnnl/dnnl.hpp"

namespace dnnl {

using dt = memory::data_type;

TEST(primitive_cache_test, TestInitState) {
    const int nnze = 12;

    memory::desc md;
    // CSR.
    ASSERT_NO_THROW(md = memory::desc({64, 128}, dt::s8,
                            memory::desc::csr(nnze, dt::s8, dt::s32)));
    // Size of values.
    ASSERT_EQ(md.get_size(0), nnze * memory::data_type_size(md.data_type()));
    // Size of indices.
    ASSERT_EQ(md.get_size(1), nnze * memory::data_type_size(dt::s8));
    // Size of pointers.
    ASSERT_EQ(md.get_size(2),
            (md.dims()[0] + 1) * memory::data_type_size(dt::s32));

    // CSC.
    ASSERT_NO_THROW(md = memory::desc({64, 128}, dt::f32,
                            memory::desc::csc(nnze, dt::s32, dt::s32)));
    // Size of values.
    ASSERT_EQ(md.get_size(0), nnze * memory::data_type_size(md.data_type()));
    // Size of indices.
    ASSERT_EQ(md.get_size(1), nnze * memory::data_type_size(dt::s32));
    // Size of pointers.
    ASSERT_EQ(md.get_size(2),
            (md.dims()[1] + 1) * memory::data_type_size(dt::s32));

    // BCSR.
    ASSERT_NO_THROW(
            md = memory::desc({64, 128}, dt::s8,
                    memory::desc::bcsr(nnze, dt::s8, dt::s32, {4, 16})));
    // Size of values.
    ASSERT_EQ(md.get_size(0),
            nnze * 4 * 16 * memory::data_type_size(md.data_type()));
    // Size of indices.
    ASSERT_EQ(md.get_size(1), nnze * memory::data_type_size(dt::s8));
    // Size of pointers.
    ASSERT_EQ(md.get_size(2),
            (md.dims()[0] + 1) * memory::data_type_size(dt::s32));

    // BCSC.
    ASSERT_NO_THROW(
            md = memory::desc({64, 128}, dt::f32,
                    memory::desc::bcsc(nnze, dt::s32, dt::s32, {4, 16})));
    // Size of values.
    ASSERT_EQ(md.get_size(0),
            nnze * 4 * 16 * memory::data_type_size(md.data_type()));
    // Size of indices.
    ASSERT_EQ(md.get_size(1), nnze * memory::data_type_size(dt::s32));
    // Size of pointers.
    ASSERT_EQ(md.get_size(2),
            (md.dims()[1] + 1) * memory::data_type_size(dt::s32));
}

} // namespace dnnl
