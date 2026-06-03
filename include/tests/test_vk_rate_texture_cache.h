// Copyright (c) 2026, Swung0x48 <swung0x48@outlook.com>
// SPDX-License-Identifier: MIT

#ifndef TEST_VK_RATE_TEXTURE_CACHE_H
#define TEST_VK_RATE_TEXTURE_CACHE_H

#ifdef __cplusplus
extern "C" {
#endif

#define TESTS_VULKAN_RATE_TEXTURE_CACHE_VERSION    TEST_MKVERSION(1, 0, 0)
#define TESTS_VULKAN_RATE_TEXTURE_CACHE_NAME       "vk_rate_texture_cache"
#define TESTS_VULKAN_RATE_TEXTURE_CACHE_LINEAR_NAME "vk_rate_texture_cache_linear"

test_status TestsVulkanRateTextureCacheRegister();

#ifdef __cplusplus
}
#endif
#endif
