// SPDX-License-Identifier: MIT

#ifndef TEST_VK_CUPROBE_H
#define TEST_VK_CUPROBE_H

#ifdef __cplusplus
extern "C" {
#endif

#define TESTS_VULKAN_CUPROBE_VERSION  TEST_MKVERSION(1, 0, 0)
#define TESTS_VULKAN_CUPROBE_NAME     "vk_cuprobe"

test_status TestsVulkanCUProbeRegister();

#ifdef __cplusplus
}
#endif
#endif
