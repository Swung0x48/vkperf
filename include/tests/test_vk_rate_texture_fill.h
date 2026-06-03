// Copyright (c) 2026, Swung0x48 <swung0x48@outlook.com>
// SPDX-License-Identifier: MIT

#ifndef TEST_VK_RATE_TEXTURE_FILL_H
#define TEST_VK_RATE_TEXTURE_FILL_H

#ifdef __cplusplus
extern "C" {
#endif

#define TESTS_VULKAN_RATE_TEXTURE_FILL_VERSION                 TEST_MKVERSION(1, 0, 0)
#define TESTS_VULKAN_RATE_TEXTURE_FILL_TEXEL_FETCH_NAME        "vk_rate_texture_fill_texel_fetch"
#define TESTS_VULKAN_RATE_TEXTURE_FILL_NEAREST_NAME            "vk_rate_texture_fill_nearest"

test_status TestsVulkanRateTextureFillRegister();

#ifdef __cplusplus
}
#endif
#endif
