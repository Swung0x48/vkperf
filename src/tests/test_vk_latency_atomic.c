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
#include "tests/test_vk_latency_atomic.h"

#define VULKAN_LATENCY_ATOMIC_TARGET_TIME_US       (250000)
#define VULKAN_LATENCY_ATOMIC_STARTING_LOOPS       (1024)
#define VULKAN_LATENCY_ATOMIC_BUFFER_SIZE          (2 * sizeof(uint32_t))
#define VULKAN_LATENCY_ATOMIC_WORKGROUPS           (2)
#define VULKAN_LATENCY_ATOMIC_HANDOFFS_PER_LOOP    (2)

typedef struct vulkan_latency_atomic_uniform_buffer_t {
    uint32_t loop_count;
} vulkan_latency_atomic_uniform_buffer;

static test_status _VulkanLatencyAtomicEntry(vulkan_physical_device *device, void *config_data);
static test_status _VulkanLatencyAtomicExecuteKernel(uint32_t loop_count, uint64_t *time_taken, vulkan_region *atomic_region, vulkan_region *uniform_region, vulkan_command_sequence *command_sequence, vulkan_compute_pipeline *pipeline);

test_status TestsVulkanLatencyAtomicRegister() {
    return VulkanRunnerRegisterTest(&_VulkanLatencyAtomicEntry, NULL, TESTS_VULKAN_LATENCY_ATOMIC_NAME, TESTS_VULKAN_LATENCY_ATOMIC_VERSION, false);
}

static test_status _VulkanLatencyAtomicEntry(vulkan_physical_device *physical_device, void *config_data) {
    TEST_UNUSED(config_data);
    test_status status = TEST_OK;

    INFO("Device Name: %s\n", physical_device->physical_properties.properties.deviceName);

    VkQueueFamilyProperties *queue_family_properties = NULL;
    uint32_t queue_family_count = 0;
    status = VulkanGetPhysicalQueueFamilyProperties(physical_device, &queue_family_properties, &queue_family_count);
    if (!TEST_SUCCESS(status)) {
        goto error;
    }

    vulkan_device device;
    status = VulkanCreateDevice(physical_device, queue_family_properties, queue_family_count, VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT, &device, NULL);
    if (!TEST_SUCCESS(status)) {
        goto cleanup_queue_properties;
    }

    vulkan_shader shader;
    status = VulkanShaderInitializeFromFile(&device, "vulkan_latency_atomic.spv", VK_SHADER_STAGE_COMPUTE_BIT, &shader);
    if (!TEST_SUCCESS(status)) {
        goto cleanup_device;
    }
    status = VulkanShaderAddDescriptor(&shader, "atomic buffer", VULKAN_BINDING_STORAGE, 0, 0);
    if (!TEST_SUCCESS(status)) {
        goto cleanup_shader;
    }
    status = VulkanShaderAddFixedSizeDescriptor(&shader, sizeof(vulkan_latency_atomic_uniform_buffer), "uniform buffer", VULKAN_BINDING_UNIFORM, 0, 1);
    if (!TEST_SUCCESS(status)) {
        goto cleanup_shader;
    }
    status = VulkanShaderCreateDescriptorSets(&shader);
    if (!TEST_SUCCESS(status)) {
        goto cleanup_shader;
    }

    vulkan_compute_pipeline pipeline;
    status = VulkanComputePipelineInitialize(&shader, "main", &pipeline);
    if (!TEST_SUCCESS(status)) {
        goto cleanup_shader;
    }

    vulkan_memory memory;
    status = VulkanMemoryInitialize(&device, VULKAN_MEMORY_NORMAL, &memory);
    if (!TEST_SUCCESS(status)) {
        goto cleanup_pipeline;
    }
    status = VulkanMemoryAddRegion(&memory, VULKAN_LATENCY_ATOMIC_BUFFER_SIZE, "atomic buffer", VULKAN_REGION_STORAGE, VULKAN_REGION_TRANSFER_DESTINATION);
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
    status = VulkanMemoryAddRegion(&uniform_memory, sizeof(vulkan_latency_atomic_uniform_buffer), "uniform buffer", VULKAN_REGION_UNIFORM, VULKAN_REGION_NORMAL);
    if (!TEST_SUCCESS(status)) {
        goto cleanup_uniform_memory;
    }
    status = VulkanMemoryAllocateBacking(&uniform_memory);
    if (!TEST_SUCCESS(status)) {
        goto cleanup_uniform_memory;
    }

    status = VulkanComputePipelineBind(&pipeline, &memory, "atomic buffer");
    if (!TEST_SUCCESS(status)) {
        goto free_uniform_memory;
    }
    status = VulkanComputePipelineBind(&pipeline, &uniform_memory, "uniform buffer");
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

    vulkan_region *atomic_region = VulkanMemoryGetRegion(&memory, "atomic buffer");
    vulkan_region *uniform_region = VulkanMemoryGetRegion(&uniform_memory, "uniform buffer");

    INFO("Warming up...\n");
    status = _VulkanLatencyAtomicExecuteKernel(VULKAN_LATENCY_ATOMIC_STARTING_LOOPS, NULL, atomic_region, uniform_region, &command_sequence, &pipeline);
    if (!TEST_SUCCESS(status)) {
        goto cleanup_command_sequence;
    }
    INFO("Warmup finished\n");

    uint32_t loop_count = VULKAN_LATENCY_ATOMIC_STARTING_LOOPS;
    uint64_t time_taken = 0;
    while (time_taken < VULKAN_LATENCY_ATOMIC_TARGET_TIME_US) {
        status = _VulkanLatencyAtomicExecuteKernel(loop_count, &time_taken, atomic_region, uniform_region, &command_sequence, &pipeline);
        if (!TEST_SUCCESS(status)) {
            goto cleanup_command_sequence;
        }
        if (time_taken < VULKAN_LATENCY_ATOMIC_TARGET_TIME_US) {
            loop_count *= 2;
        }
    }

    uint64_t total_handoffs = (uint64_t)loop_count * VULKAN_LATENCY_ATOMIC_HANDOFFS_PER_LOOP;
    double latency_ns = ((double)time_taken * 1000.0) / (double)total_handoffs;

    INFO("Final results for %s:\n", physical_device->physical_properties.properties.deviceName);
    if (MainGetTestResultFormat() == test_result_csv) {
        LOG_PLAIN("%s,\n", physical_device->physical_properties.properties.deviceName);
        LOG_PLAIN("Atomic Pingpong Latency (ns),Loops,Handoffs,Workgroups\n");
        LOG_PLAIN("%.3f,%lu,%llu,%u\n", latency_ns, loop_count, total_handoffs, VULKAN_LATENCY_ATOMIC_WORKGROUPS);
    } else if (MainGetTestResultFormat() == test_result_raw) {
        LOG_RESULT(0, "%u", "%llu", 0, (uint64_t)(latency_ns * 1000.0));
    } else {
        INFO("Atomic pingpong latency: %.3fns (loops: %lu, handoffs: %llu, workgroups: %u)\n", latency_ns, loop_count, total_handoffs, VULKAN_LATENCY_ATOMIC_WORKGROUPS);
    }

cleanup_command_sequence:
    VulkanCommandBufferReset(&command_sequence);
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
cleanup_pipeline:
    VulkanComputePipelineCleanUp(&pipeline);
cleanup_shader:
    VulkanShaderCleanUp(&shader);
cleanup_device:
    VulkanDestroyDevice(&device);
cleanup_queue_properties:
    free(queue_family_properties);
error:
    return status;
}

static test_status _VulkanLatencyAtomicExecuteKernel(uint32_t loop_count, uint64_t *time_taken, vulkan_region *atomic_region, vulkan_region *uniform_region, vulkan_command_sequence *command_sequence, vulkan_compute_pipeline *pipeline) {
    test_status status = BufferFillerZero(atomic_region);
    if (!TEST_SUCCESS(status)) {
        goto error;
    }

    volatile vulkan_latency_atomic_uniform_buffer *uniform_buffer_memory = VulkanMemoryMap(uniform_region);
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
    status = VulkanCommandBufferDispatch(command_sequence, VULKAN_LATENCY_ATOMIC_WORKGROUPS, 1, 1);
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
        INFO("Loop count %lu took %.3fms (atomic pingpong latency: N/A)\n", loop_count, time / 1000.0f);
    } else {
        uint64_t total_handoffs = (uint64_t)loop_count * VULKAN_LATENCY_ATOMIC_HANDOFFS_PER_LOOP;
        double latency_ns = ((double)time * 1000.0) / (double)total_handoffs;
        INFO("Loop count %lu took %.3fms (atomic pingpong latency: %.3fns, handoffs: %llu)\n", loop_count, time / 1000.0f, latency_ns, total_handoffs);
    }

cleanup_command_sequence:
    VulkanCommandBufferReset(command_sequence);
error:
    return status;
}
