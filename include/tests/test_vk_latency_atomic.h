// Copyright (c) 2026, Swung0x48 <swung0x48@outlook.com>
// SPDX-License-Identifier: MIT

#ifndef TEST_VK_LATENCY_ATOMIC_H
#define TEST_VK_LATENCY_ATOMIC_H

#ifdef __cplusplus
extern "C" {
#endif

#define TESTS_VULKAN_LATENCY_ATOMIC_VERSION    TEST_MKVERSION(1, 0, 0)
#define TESTS_VULKAN_LATENCY_ATOMIC_NAME       "vk_latency_atomic"

test_status TestsVulkanLatencyAtomicRegister();

#ifdef __cplusplus
}
#endif
#endif
