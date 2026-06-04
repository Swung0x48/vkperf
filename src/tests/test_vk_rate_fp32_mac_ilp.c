// Copyright (c) 2026, Swung0x48 <swung0x48@outlook.com>
// SPDX-License-Identifier: MIT

#include "main.h"
#include "logger.h"
#include "vulkan_helper.h"
#include "vulkan_runner.h"
#include "vulkan_memory.h"
#include "vulkan_shader.h"
#include "vulkan_compute_pipeline.h"
#include "vulkan_command_buffer.h"
#include "buffer_filler.h"
#include "tests/test_vk_rate_fp32_mac_ilp.h"

#define VULKAN_RATE_FP32_MAC_ILP_SCALARS_PER_THREAD     (16)
#define VULKAN_RATE_FP32_MAC_ILP_WORKGROUP_SIZE         (256)
#define VULKAN_RATE_FP32_MAC_ILP_STARTING_WORKGROUPS    (16)
#define VULKAN_RATE_FP32_MAC_ILP_STARTING_LOOPS         (1024)
#define VULKAN_RATE_FP32_MAC_ILP_TARGET_TIME_US         (250000)

typedef struct vulkan_rate_fp32_mac_ilp_uniform_buffer_t {
    uint32_t loop_count;
} vulkan_rate_fp32_mac_ilp_uniform_buffer;

typedef struct vulkan_rate_fp32_mac_ilp_case_t {
    uint32_t ilp;
    uint32_t fmas_per_loop;
} vulkan_rate_fp32_mac_ilp_case;

static const vulkan_rate_fp32_mac_ilp_case _vulkan_rate_fp32_mac_ilp_cases[] = {
    { 1, 16 },
    { 2, 16 },
    { 3, 15 },
    { 4, 16 },
    { 8, 16 },
    { 16, 16 },
};

static test_status _VulkanRateFp32MacIlpEntry(vulkan_physical_device *physical_device, void *config_data);
static test_status _VulkanRateFp32MacIlpRunCase(const vulkan_rate_fp32_mac_ilp_case *test_case, vulkan_device *device, vulkan_shader *shader, vulkan_memory *memory, vulkan_memory *uniform_memory, vulkan_command_sequence *command_sequence);
static test_status _VulkanRateFp32MacIlpExecuteKernel(uint64_t workgroup_count, uint32_t loop_count, uint32_t fmas_per_loop, uint64_t *result, uint64_t *time_taken, vulkan_device *device, vulkan_region *uniform_region, vulkan_command_sequence *command_sequence, vulkan_compute_pipeline *pipeline);

test_status TestsVulkanRateFp32MacIlpRegister() {
    return VulkanRunnerRegisterTest(&_VulkanRateFp32MacIlpEntry, NULL, TESTS_VULKAN_RATE_FP32_MAC_ILP_NAME, TESTS_VULKAN_RATE_FP32_MAC_ILP_VERSION, false);
}

static test_status _VulkanRateFp32MacIlpEntry(vulkan_physical_device *physical_device, void *config_data) {
    (void)config_data;
    test_status status = TEST_OK;

    INFO("Device Name: %s\n", physical_device->physical_properties.properties.deviceName);

    VkQueueFamilyProperties *queue_family_properties = NULL;
    uint32_t queue_family_count = 0;
    status = VulkanGetPhysicalQueueFamilyProperties(physical_device, &queue_family_properties, &queue_family_count);
    if (!TEST_SUCCESS(status)) {
        goto error;
    }

    VkPhysicalDeviceFeatures2 enabled_features = {0};
    enabled_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    enabled_features.pNext = NULL;

    vulkan_device device;
    status = VulkanCreateDevice(physical_device, queue_family_properties, queue_family_count, VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT, &device, &enabled_features);
    if (!TEST_SUCCESS(status)) {
        goto cleanup_queue_properties;
    }

    vulkan_shader shader;
    status = VulkanShaderInitializeFromFile(&device, "vulkan_rate_fp32_mac_ilp.spv", VK_SHADER_STAGE_COMPUTE_BIT, &shader);
    if (!TEST_SUCCESS(status)) {
        goto cleanup_device;
    }
    status = VulkanShaderAddDescriptor(&shader, "dummy inputs", VULKAN_BINDING_STORAGE, 0, 0);
    if (!TEST_SUCCESS(status)) {
        goto cleanup_shader;
    }
    status = VulkanShaderAddDescriptor(&shader, "dummy outputs", VULKAN_BINDING_STORAGE, 0, 1);
    if (!TEST_SUCCESS(status)) {
        goto cleanup_shader;
    }
    status = VulkanShaderAddFixedSizeDescriptor(&shader, sizeof(vulkan_rate_fp32_mac_ilp_uniform_buffer), "uniform buffer", VULKAN_BINDING_UNIFORM, 0, 2);
    if (!TEST_SUCCESS(status)) {
        goto cleanup_shader;
    }
    status = VulkanShaderCreateDescriptorSets(&shader);
    if (!TEST_SUCCESS(status)) {
        goto cleanup_shader;
    }

    vulkan_memory memory;
    status = VulkanMemoryInitialize(&device, VULKAN_MEMORY_NORMAL, &memory);
    if (!TEST_SUCCESS(status)) {
        goto cleanup_shader;
    }
    uint64_t dummy_region_size = VULKAN_RATE_FP32_MAC_ILP_WORKGROUP_SIZE * VULKAN_RATE_FP32_MAC_ILP_SCALARS_PER_THREAD * sizeof(float);
    INFO("Allocating %llu bytes of dummy data\n", dummy_region_size);
    status = VulkanMemoryAddRegion(&memory, dummy_region_size, "dummy inputs", VULKAN_REGION_STORAGE, VULKAN_REGION_TRANSFER_DESTINATION);
    if (!TEST_SUCCESS(status)) {
        goto cleanup_memory;
    }
    status = VulkanMemoryAddRegion(&memory, dummy_region_size, "dummy outputs", VULKAN_REGION_STORAGE, VULKAN_REGION_TRANSFER_DESTINATION);
    if (!TEST_SUCCESS(status)) {
        goto cleanup_memory;
    }
    status = VulkanMemoryAllocateBacking(&memory);
    if (!TEST_SUCCESS(status)) {
        goto cleanup_memory;
    }

    vulkan_memory uniform_memory;
    status = VulkanMemoryInitialize(&device, VULKAN_MEMORY_VISIBLE | VULKAN_MEMORY_HOST_LOCAL, &uniform_memory);
    if (!TEST_SUCCESS(status)) {
        goto free_memory;
    }
    status = VulkanMemoryAddRegion(&uniform_memory, sizeof(vulkan_rate_fp32_mac_ilp_uniform_buffer), "uniform buffer", VULKAN_REGION_UNIFORM, VULKAN_REGION_NORMAL);
    if (!TEST_SUCCESS(status)) {
        goto cleanup_uniform_memory;
    }
    status = VulkanMemoryAllocateBacking(&uniform_memory);
    if (!TEST_SUCCESS(status)) {
        goto cleanup_uniform_memory;
    }

    vulkan_region *data_region = VulkanMemoryGetRegion(&memory, "dummy inputs");
    status = BufferFillerValueFloat(data_region, 1.0f);
    if (!TEST_SUCCESS(status)) {
        goto free_uniform_memory;
    }

    vulkan_command_buffer command_buffer;
    status = VulkanCommandBufferInitialize(&device, 1, &command_buffer);
    if (!TEST_SUCCESS(status)) {
        goto free_uniform_memory;
    }
    vulkan_command_sequence command_sequence;
    status = VulkanCommandSequenceInitialize(&command_buffer, VULKAN_COMMAND_SEQUENCE_RESET_ON_COMPLETION, &command_sequence);
    if (!TEST_SUCCESS(status)) {
        goto cleanup_command_buffer;
    }

    if (MainGetTestResultFormat() == test_result_csv) {
        LOG_PLAIN("%s,\n", physical_device->physical_properties.properties.deviceName);
        LOG_PLAIN("ILP,FFMA Instruction Rate (GInstr/s),FP32 MAC Rate (GFLOPS),Workgroups,Loops,Scalar FMA/Loop\n");
    }

    size_t case_count = sizeof(_vulkan_rate_fp32_mac_ilp_cases) / sizeof(_vulkan_rate_fp32_mac_ilp_cases[0]);
    for (size_t i = 0; i < case_count; i++) {
        status = _VulkanRateFp32MacIlpRunCase(&_vulkan_rate_fp32_mac_ilp_cases[i], &device, &shader, &memory, &uniform_memory, &command_sequence);
        if (!TEST_SUCCESS(status)) {
            goto cleanup_command_buffer;
        }
    }

cleanup_command_buffer:
    VulkanCommandBufferCleanUp(&command_buffer);
free_uniform_memory:
    VulkanMemoryFreeBuffersAndBacking(&uniform_memory);
cleanup_uniform_memory:
    VulkanMemoryCleanUp(&uniform_memory);
free_memory:
    VulkanMemoryFreeBuffersAndBacking(&memory);
cleanup_memory:
    VulkanMemoryCleanUp(&memory);
cleanup_shader:
    VulkanShaderCleanUp(&shader);
cleanup_device:
    VulkanDestroyDevice(&device);
cleanup_queue_properties:
    free(queue_family_properties);
error:
    return status;
}

static test_status _VulkanRateFp32MacIlpRunCase(const vulkan_rate_fp32_mac_ilp_case *test_case, vulkan_device *device, vulkan_shader *shader, vulkan_memory *memory, vulkan_memory *uniform_memory, vulkan_command_sequence *command_sequence) {
    test_status status = TEST_OK;

    VkSpecializationMapEntry specialization_entry = {0};
    specialization_entry.constantID = 0;
    specialization_entry.offset = 0;
    specialization_entry.size = sizeof(test_case->ilp);

    VkSpecializationInfo specialization_info = {0};
    specialization_info.mapEntryCount = 1;
    specialization_info.pMapEntries = &specialization_entry;
    specialization_info.dataSize = sizeof(test_case->ilp);
    specialization_info.pData = &test_case->ilp;

    vulkan_compute_pipeline pipeline;
    status = VulkanComputePipelineInitializeSpecialized(shader, "main", &specialization_info, &pipeline);
    if (!TEST_SUCCESS(status)) {
        goto error;
    }
    status = VulkanComputePipelineBind(&pipeline, memory, "dummy inputs");
    if (!TEST_SUCCESS(status)) {
        goto cleanup_pipeline;
    }
    status = VulkanComputePipelineBind(&pipeline, memory, "dummy outputs");
    if (!TEST_SUCCESS(status)) {
        goto cleanup_pipeline;
    }
    status = VulkanComputePipelineBind(&pipeline, uniform_memory, "uniform buffer");
    if (!TEST_SUCCESS(status)) {
        goto cleanup_pipeline;
    }

    vulkan_region *uniform_region = VulkanMemoryGetRegion(uniform_memory, "uniform buffer");
    INFO("Warming up ILP %u...\n", test_case->ilp);
    for (int i = 0; i < 5; i++) {
        status = _VulkanRateFp32MacIlpExecuteKernel(VULKAN_RATE_FP32_MAC_ILP_STARTING_WORKGROUPS, VULKAN_RATE_FP32_MAC_ILP_STARTING_LOOPS, test_case->fmas_per_loop, NULL, NULL, device, uniform_region, command_sequence, &pipeline);
        if (!TEST_SUCCESS(status)) {
            goto cleanup_pipeline;
        }
    }

    uint64_t top_result = 0;
    uint64_t top_loops = 0;
    uint64_t top_workgroups = 0;
    uint64_t workgroup_count = VULKAN_RATE_FP32_MAC_ILP_STARTING_WORKGROUPS;

    while (true) {
        uint64_t result = 0;
        uint64_t time_taken = 0;
        uint32_t loop_count = VULKAN_RATE_FP32_MAC_ILP_STARTING_LOOPS;

        while (time_taken < VULKAN_RATE_FP32_MAC_ILP_TARGET_TIME_US) {
            status = _VulkanRateFp32MacIlpExecuteKernel(workgroup_count, loop_count, test_case->fmas_per_loop, &result, &time_taken, device, uniform_region, command_sequence, &pipeline);
            if (!TEST_SUCCESS(status)) {
                goto cleanup_pipeline;
            }
            if (result > top_result) {
                top_result = result;
                top_loops = loop_count;
                top_workgroups = workgroup_count;
            }
            loop_count *= 2;
        }
        if (loop_count == VULKAN_RATE_FP32_MAC_ILP_STARTING_LOOPS * 2) {
            break;
        }
        workgroup_count *= 2;
    }

    if (MainGetTestResultFormat() == test_result_csv) {
        LOG_PLAIN("%u,%.3f,%.3f,%llu,%llu,%u\n",
            test_case->ilp,
            (double)top_result / 1000000000.0,
            (double)(top_result * 2) / 1000000000.0,
            top_workgroups,
            top_loops,
            test_case->fmas_per_loop);
    } else if (MainGetTestResultFormat() == test_result_raw) {
        LOG_RESULT(0, "%u", "%llu", test_case->ilp, top_result);
    } else {
        helper_unit_pair instruction_conversion;
        helper_unit_pair flop_conversion;
        HelperConvertUnitsPlain1000(top_result, &instruction_conversion);
        HelperConvertUnitsPlain1000(top_result * 2, &flop_conversion);
        INFO("ILP %u: %.3f %sFFMA/s, %.3f %sFLOPS (workgroups: %llu, loops: %llu)\n",
            test_case->ilp,
            instruction_conversion.value,
            instruction_conversion.units,
            flop_conversion.value,
            flop_conversion.units,
            top_workgroups,
            top_loops);
    }

cleanup_pipeline:
    VulkanComputePipelineCleanUp(&pipeline);
error:
    return status;
}

static test_status _VulkanRateFp32MacIlpExecuteKernel(uint64_t workgroup_count, uint32_t loop_count, uint32_t fmas_per_loop, uint64_t *result, uint64_t *time_taken, vulkan_device *device, vulkan_region *uniform_region, vulkan_command_sequence *command_sequence, vulkan_compute_pipeline *pipeline) {
    helper_unit_pair unit_conversion;
    test_status status = TEST_OK;
    volatile vulkan_rate_fp32_mac_ilp_uniform_buffer *uniform_buffer_memory = VulkanMemoryMap(uniform_region);
    if (uniform_buffer_memory == NULL) {
        status = TEST_INVALID_PARAMETER;
        goto error;
    }
    uniform_buffer_memory->loop_count = loop_count;
    VulkanMemoryUnmap(uniform_region);

    status = VulkanCommandBufferStart(command_sequence);
    if (!TEST_SUCCESS(status)) {
        goto error;
    }
    status = VulkanCommandBufferBindComputePipeline(command_sequence, pipeline);
    if (!TEST_SUCCESS(status)) {
        goto cleanup_command_sequence;
    }
    uint32_t groups_x = 0;
    uint32_t groups_y = 0;
    uint32_t groups_z = 0;
    status = VulkanCalculateWorkgroupDispatch(device, workgroup_count, &groups_x, &groups_y, &groups_z);
    if (!TEST_SUCCESS(status)) {
        goto cleanup_command_sequence;
    }
    status = VulkanCommandBufferDispatch(command_sequence, groups_x, groups_y, groups_z);
    if (!TEST_SUCCESS(status)) {
        goto cleanup_command_sequence;
    }
    status = VulkanCommandBufferEnd(command_sequence);
    if (!TEST_SUCCESS(status)) {
        goto cleanup_command_sequence;
    }
    HelperResetTimestamp();
    status = VulkanCommandBufferSubmit(command_sequence);
    if (!TEST_SUCCESS(status)) {
        goto cleanup_command_sequence;
    }
    status = VulkanCommandBufferWait(command_sequence, VULKAN_COMMAND_SEQUENCE_WAIT_INFINITE);
    if (!TEST_SUCCESS(status)) {
        goto cleanup_command_sequence;
    }
    uint64_t time = HelperMarkTimestamp();
    if (time_taken != NULL) {
        *time_taken = time;
    }
    if (time == 0) {
        INFO("Loop count %lu workgroup count %llu took %.3fms (rate: N/A)\n", loop_count, workgroup_count, time / 1000.0f);
        if (result != NULL) {
            *result = 0;
        }
    } else {
        uint64_t total_instructions = (uint64_t)groups_x * (uint64_t)groups_y * (uint64_t)groups_z * VULKAN_RATE_FP32_MAC_ILP_WORKGROUP_SIZE * (uint64_t)fmas_per_loop * (uint64_t)loop_count;
        uint64_t throughput_per_second = (total_instructions * 1000000) / time;
        HelperConvertUnitsPlain1000(throughput_per_second, &unit_conversion);
        INFO("Loop count %lu workgroup count %llu took %.3fms (rate: %.3f %sFFMA/s)\n", loop_count, workgroup_count, time / 1000.0f, unit_conversion.value, unit_conversion.units);
        if (result != NULL) {
            *result = throughput_per_second;
        }
    }
cleanup_command_sequence:
    VulkanCommandBufferReset(command_sequence);
error:
    return status;
}
