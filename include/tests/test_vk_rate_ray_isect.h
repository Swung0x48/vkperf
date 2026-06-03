// Copyright (c) 2026, Swung0x48 <swung0x48@outlook.com>
// SPDX-License-Identifier: MIT

#ifndef TEST_VK_RATE_RAY_ISECT_H
#define TEST_VK_RATE_RAY_ISECT_H

#ifdef __cplusplus
extern "C" {
#endif

#define TESTS_VULKAN_RATE_RAY_ISECT_VERSION          TEST_MKVERSION(1, 2, 0)
#define TESTS_VULKAN_RATE_RAYBOX_ISECT_NAME          "vk_rate_raybox_isect"
#define TESTS_VULKAN_RATE_RAYTRI_ISECT_NAME          "vk_rate_raytri_isect"
#define TESTS_VULKAN_RATE_RAYBOX_ISECT_ALL_HIT_NAME  "vk_rate_raybox_isect_all_hit"
#define TESTS_VULKAN_RATE_RAYBOX_ISECT_ALL_MISS_NAME "vk_rate_raybox_isect_all_miss"
#define TESTS_VULKAN_RATE_RAYBOX_ISECT_RANDOM_NAME   "vk_rate_raybox_isect_random"
#define TESTS_VULKAN_RATE_RAYTRI_ISECT_ALL_HIT_NAME  "vk_rate_raytri_isect_all_hit"
#define TESTS_VULKAN_RATE_RAYTRI_ISECT_ALL_MISS_NAME "vk_rate_raytri_isect_all_miss"
#define TESTS_VULKAN_RATE_RAYTRI_ISECT_RANDOM_NAME   "vk_rate_raytri_isect_random"

test_status TestsVulkanRateRayIsectRegister();

#ifdef __cplusplus
}
#endif
#endif
