// Copyright (c) 2026, Swung0x48 <swung0x48@outlook.com>
// SPDX-License-Identifier: MIT

#include "main.h"
#include "logger.h"
#include "helper.h"
#include "vulkan_helper.h"
#include "vulkan_runner.h"
#include "vulkan_memory.h"
#include "vulkan_shader.h"
#include "vulkan_compute_pipeline.h"
#include "vulkan_command_buffer.h"
#include "vulkan_texture.h"
#include "vulkan_staging.h"
#include "tests/test_vk_rate_texture_cache.h"

#define VULKAN_RATE_TEXTURE_CACHE_PARALLEL_FETCHES       (16)
#define VULKAN_RATE_TEXTURE_CACHE_WORKGROUP_SIZE         (256)
#define VULKAN_RATE_TEXTURE_CACHE_STARTING_WORKGROUPS    (16)
#define VULKAN_RATE_TEXTURE_CACHE_STARTING_LOOP_COUNT    (128)
#define VULKAN_RATE_TEXTURE_CACHE_TARGET_TIME_US         (150000)
#define VULKAN_RATE_TEXTURE_CACHE_WARMUP_RUNS            (3)
#define VULKAN_RATE_TEXTURE_CACHE_MAX_TEXTURE_SIDE       (8192)
#define VULKAN_RATE_TEXTURE_CACHE_RNG_SEED               (1140071481)
#define VULKAN_RATE_TEXTURE_CACHE_OP_TEXEL_FETCH         (0)
#define VULKAN_RATE_TEXTURE_CACHE_OP_LINEAR              (1)

typedef struct vulkan_rate_texture_cache_uniform_buffer_t {
    uint32_t loop_count;
    uint32_t texture_x_mask;
    uint32_t texture_pixel_mask;
    uint32_t texture_side_shift;
    float inv_texture_side;
    uint32_t padding[3];
} vulkan_rate_texture_cache_uniform_buffer;

static test_status _VulkanRateTextureCacheEntry(vulkan_physical_device *device, void *config_data);
static test_status _VulkanRateTextureCacheRegisterSubtest(uint32_t operation, const char *test_name);
static test_status _VulkanRateTextureCacheRunSide(vulkan_device *device, vulkan_compute_pipeline *pipeline, vulkan_memory *uniform_memory, vulkan_command_sequence *command_sequence, uint32_t texture_side, uint32_t operation, uint64_t *top_result, uint64_t *top_workgroups, uint64_t *top_loops);
static test_status _VulkanRateTextureCacheFillTexture(vulkan_texture *texture, uint32_t texture_side);
static test_status _VulkanRateTextureCacheConfigureSampler(vulkan_texture *texture, uint32_t operation);
static test_status _VulkanRateTextureCacheExecuteKernel(uint64_t workgroup_count, uint32_t loop_count, uint32_t texture_side, uint64_t *result, uint64_t *time_taken, vulkan_device *device, vulkan_region *uniform_region, vulkan_command_sequence *command_sequence, vulkan_compute_pipeline *pipeline);
static uint32_t _VulkanRateTextureCachePowerOfTwoLog2(uint32_t value);
static const char *_VulkanRateTextureCacheGetShaderName(uint32_t operation);
static const char *_VulkanRateTextureCacheGetOperationName(uint32_t operation);

test_status TestsVulkanRateTextureCacheRegister() {
    test_status status = _VulkanRateTextureCacheRegisterSubtest(VULKAN_RATE_TEXTURE_CACHE_OP_TEXEL_FETCH, TESTS_VULKAN_RATE_TEXTURE_CACHE_NAME);
    TEST_RETFAIL(status);
    return _VulkanRateTextureCacheRegisterSubtest(VULKAN_RATE_TEXTURE_CACHE_OP_LINEAR, TESTS_VULKAN_RATE_TEXTURE_CACHE_LINEAR_NAME);
}

static test_status _VulkanRateTextureCacheRegisterSubtest(uint32_t operation, const char *test_name) {
    return VulkanRunnerRegisterTest(&_VulkanRateTextureCacheEntry, (void *)(uint64_t)operation, test_name, TESTS_VULKAN_RATE_TEXTURE_CACHE_VERSION, false);
}

static test_status _VulkanRateTextureCacheEntry(vulkan_physical_device *physical_device, void *config_data) {
    test_status status = TEST_OK;
    uint32_t operation = (uint32_t)(uint64_t)config_data;
    const char *shader_name = _VulkanRateTextureCacheGetShaderName(operation);
    const char *operation_name = _VulkanRateTextureCacheGetOperationName(operation);
    if (shader_name == NULL || operation_name == NULL) {
        return TEST_PROGRAMMING_ERROR;
    }

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

    uint32_t maximum_texture_side = min(device.physical_device->physical_properties.properties.limits.maxImageDimension2D, VULKAN_RATE_TEXTURE_CACHE_MAX_TEXTURE_SIDE);
    INFO("Maximum texture side tested: %lu\n", maximum_texture_side);

    vulkan_shader shader;
    status = VulkanShaderInitializeFromFile(&device, shader_name, VK_SHADER_STAGE_COMPUTE_BIT, &shader);
    if (!TEST_SUCCESS(status)) {
        goto cleanup_device;
    }
    status = VulkanShaderAddDescriptor(&shader, "source texture", VULKAN_BINDING_SAMPLER, 0, 0);
    if (!TEST_SUCCESS(status)) {
        goto cleanup_shader;
    }
    status = VulkanShaderAddDescriptor(&shader, "dummy outputs", VULKAN_BINDING_STORAGE, 0, 1);
    if (!TEST_SUCCESS(status)) {
        goto cleanup_shader;
    }
    status = VulkanShaderAddFixedSizeDescriptor(&shader, sizeof(vulkan_rate_texture_cache_uniform_buffer), "uniform buffer", VULKAN_BINDING_UNIFORM, 0, 2);
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

    vulkan_memory uniform_memory;
    status = VulkanMemoryInitialize(&device, VULKAN_MEMORY_VISIBLE | VULKAN_MEMORY_HOST_LOCAL, &uniform_memory);
    if (!TEST_SUCCESS(status)) {
        goto cleanup_pipeline;
    }
    status = VulkanMemoryAddRegion(&uniform_memory, sizeof(vulkan_rate_texture_cache_uniform_buffer), "uniform buffer", VULKAN_REGION_UNIFORM, VULKAN_REGION_NORMAL);
    if (!TEST_SUCCESS(status)) {
        goto cleanup_uniform_memory;
    }
    status = VulkanMemoryAllocateBacking(&uniform_memory);
    if (!TEST_SUCCESS(status)) {
        goto cleanup_uniform_memory;
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

    INFO("Final results for %s:\n", physical_device->physical_properties.properties.deviceName);
    if (MainGetTestResultFormat() == test_result_csv) {
        LOG_PLAIN("%s,\n", physical_device->physical_properties.properties.deviceName);
        LOG_PLAIN("Operation,Texture Edge,Working Set (Bytes),Texture Cache Probe Rate (GTexel/s),Workgroups,Loops\n");
    }

    uint32_t result_index = 0;
    for (uint32_t texture_side = 1; texture_side <= maximum_texture_side; texture_side <<= 1) {
        uint64_t top_result = 0;
        uint64_t top_workgroups = 0;
        uint64_t top_loops = 0;

        INFO("Testing %s texture side %lu...\n", operation_name, texture_side);
        status = _VulkanRateTextureCacheRunSide(&device, &pipeline, &uniform_memory, &command_sequence, texture_side, operation, &top_result, &top_workgroups, &top_loops);
        if (!TEST_SUCCESS(status)) {
            goto cleanup_command_buffer;
        }

        uint64_t working_set_bytes = (uint64_t)texture_side * texture_side * sizeof(uint32_t);
        if (MainGetTestResultFormat() == test_result_csv) {
            LOG_PLAIN("%s,%lu,%llu,%.3f,%llu,%llu\n", operation_name, texture_side, working_set_bytes, (double)top_result / 1000000000.0, top_workgroups, top_loops);
        } else if (MainGetTestResultFormat() == test_result_raw) {
            LOG_RESULT(result_index, "%lu", "%llu", texture_side, top_result);
        } else {
            helper_unit_pair texel_conversion;
            HelperConvertUnitsPlain1000(top_result, &texel_conversion);
            INFO("Texture cache probe for %s %lux%lu (%llu bytes): %.3f %sTexel/s (workgroups: %llu, loops: %llu)\n", operation_name, texture_side, texture_side, working_set_bytes, texel_conversion.value, texel_conversion.units, top_workgroups, top_loops);
        }
        result_index++;
    }

cleanup_command_buffer:
    VulkanCommandBufferCleanUp(&command_buffer);
free_uniform_memory:
    VulkanMemoryFreeBuffersAndBacking(&uniform_memory);
cleanup_uniform_memory:
    VulkanMemoryCleanUp(&uniform_memory);
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

static test_status _VulkanRateTextureCacheRunSide(vulkan_device *device, vulkan_compute_pipeline *pipeline, vulkan_memory *uniform_memory, vulkan_command_sequence *command_sequence, uint32_t texture_side, uint32_t operation, uint64_t *top_result, uint64_t *top_workgroups, uint64_t *top_loops) {
    test_status status = TEST_OK;

    vulkan_memory memory;
    status = VulkanMemoryInitialize(device, VULKAN_MEMORY_NORMAL, &memory);
    TEST_RETFAIL(status);
    status = VulkanMemoryAddTexture2D(&memory, texture_side, texture_side, VK_FORMAT_R8G8B8A8_UNORM, 1, "source texture");
    if (!TEST_SUCCESS(status)) {
        goto cleanup_memory;
    }
    status = VulkanMemoryAddRegion(&memory, VULKAN_RATE_TEXTURE_CACHE_WORKGROUP_SIZE * sizeof(float) * 4, "dummy outputs", VULKAN_REGION_STORAGE, VULKAN_REGION_NORMAL);
    if (!TEST_SUCCESS(status)) {
        goto cleanup_memory;
    }
    status = VulkanMemoryAllocateBacking(&memory);
    if (!TEST_SUCCESS(status)) {
        goto cleanup_memory;
    }

    vulkan_texture *source_texture = VulkanMemoryGetTexture(&memory, "source texture");
    status = _VulkanRateTextureCacheConfigureSampler(source_texture, operation);
    if (!TEST_SUCCESS(status)) {
        goto free_memory;
    }
    status = _VulkanRateTextureCacheFillTexture(source_texture, texture_side);
    if (!TEST_SUCCESS(status)) {
        goto free_memory;
    }

    status = VulkanComputePipelineBind(pipeline, &memory, "source texture");
    if (!TEST_SUCCESS(status)) {
        goto free_memory;
    }
    status = VulkanComputePipelineBind(pipeline, &memory, "dummy outputs");
    if (!TEST_SUCCESS(status)) {
        goto free_memory;
    }

    vulkan_region *uniform_region = VulkanMemoryGetRegion(uniform_memory, "uniform buffer");

    INFO("Warming up texture side %lu...\n", texture_side);
    for (uint32_t i = 0; i < VULKAN_RATE_TEXTURE_CACHE_WARMUP_RUNS; i++) {
        status = _VulkanRateTextureCacheExecuteKernel(VULKAN_RATE_TEXTURE_CACHE_STARTING_WORKGROUPS, VULKAN_RATE_TEXTURE_CACHE_STARTING_LOOP_COUNT, texture_side, NULL, NULL, device, uniform_region, command_sequence, pipeline);
        if (!TEST_SUCCESS(status)) {
            goto free_memory;
        }
    }
    INFO("Warmup finished for texture side %lu\n", texture_side);

    *top_result = 0;
    *top_loops = 0;
    *top_workgroups = 0;
    uint64_t workgroup_count = VULKAN_RATE_TEXTURE_CACHE_STARTING_WORKGROUPS;

    while (true) {
        uint64_t result = 0;
        uint64_t time_taken = 0;
        uint32_t loop_count = VULKAN_RATE_TEXTURE_CACHE_STARTING_LOOP_COUNT;

        while (time_taken < VULKAN_RATE_TEXTURE_CACHE_TARGET_TIME_US) {
            status = _VulkanRateTextureCacheExecuteKernel(workgroup_count, loop_count, texture_side, &result, &time_taken, device, uniform_region, command_sequence, pipeline);
            if (!TEST_SUCCESS(status)) {
                goto free_memory;
            }
            if (result > *top_result) {
                *top_result = result;
                *top_loops = loop_count;
                *top_workgroups = workgroup_count;
            }
            loop_count *= 2;
        }
        if (loop_count == VULKAN_RATE_TEXTURE_CACHE_STARTING_LOOP_COUNT * 2) {
            break;
        }
        workgroup_count *= 2;
    }

free_memory:
    VulkanMemoryFreeBuffersAndBacking(&memory);
cleanup_memory:
    VulkanMemoryCleanUp(&memory);
    return status;
}

static test_status _VulkanRateTextureCacheFillTexture(vulkan_texture *texture, uint32_t texture_side) {
    test_status status = VulkanTexturePrepareForCopy(texture);
    TEST_RETFAIL(status);

    size_t texel_count = (size_t)texture_side * (size_t)texture_side;
    size_t texture_size = texel_count * sizeof(uint32_t);
    vulkan_staging staging;
    status = VulkanStagingInitializeImage(texture, texture_size, &staging);
    TEST_RETFAIL(status);

    uint32_t *staging_data = (uint32_t *)VulkanStagingGetBuffer(&staging);
    if (staging_data == NULL) {
        VulkanStagingCleanUp(&staging);
        return TEST_VK_MEMORY_MAPPING_ERROR;
    }

    uint64_t rng_state;
    HelperSeedRandom(&rng_state, VULKAN_RATE_TEXTURE_CACHE_RNG_SEED);
    for (size_t i = 0; i < texel_count; i++) {
        staging_data[i] = (uint32_t)HelperGenerateRandom(&rng_state);
    }

    status = VulkanStagingTransferImage(&staging, 0);
    test_status cleanup_status = VulkanStagingCleanUp(&staging);
    if (!TEST_SUCCESS(status)) {
        return status;
    }
    TEST_RETFAIL(cleanup_status);
    return VulkanTexturePrepareForRender(texture);
}

static test_status _VulkanRateTextureCacheConfigureSampler(vulkan_texture *texture, uint32_t operation) {
    if (texture == NULL) {
        return TEST_INVALID_PARAMETER;
    }
    if (operation != VULKAN_RATE_TEXTURE_CACHE_OP_LINEAR) {
        return TEST_OK;
    }

    VkDevice device = texture->memory_pool->device->device;
    if (texture->image_sampler != VK_NULL_HANDLE) {
        vkDestroySampler(device, texture->image_sampler, NULL);
        texture->image_sampler = VK_NULL_HANDLE;
    }

    VkSamplerCreateInfo sampler_create_info = {0};
    sampler_create_info.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    sampler_create_info.pNext = NULL;
    sampler_create_info.magFilter = VK_FILTER_LINEAR;
    sampler_create_info.minFilter = VK_FILTER_LINEAR;
    sampler_create_info.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    sampler_create_info.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    sampler_create_info.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    sampler_create_info.anisotropyEnable = VK_FALSE;
    sampler_create_info.maxAnisotropy = 0;
    sampler_create_info.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
    sampler_create_info.unnormalizedCoordinates = VK_FALSE;
    sampler_create_info.compareEnable = VK_FALSE;
    sampler_create_info.compareOp = VK_COMPARE_OP_ALWAYS;
    sampler_create_info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    sampler_create_info.mipLodBias = 0.0f;
    sampler_create_info.minLod = 0.0f;
    sampler_create_info.maxLod = 0.0f;

    VkResult res = vkCreateSampler(device, &sampler_create_info, NULL, &(texture->image_sampler));
    VULKAN_RETFAIL(res, TEST_VK_SAMPLER_CREATION_ERROR);
    return TEST_OK;
}

static test_status _VulkanRateTextureCacheExecuteKernel(uint64_t workgroup_count, uint32_t loop_count, uint32_t texture_side, uint64_t *result, uint64_t *time_taken, vulkan_device *device, vulkan_region *uniform_region, vulkan_command_sequence *command_sequence, vulkan_compute_pipeline *pipeline) {
    test_status status = TEST_OK;
    volatile vulkan_rate_texture_cache_uniform_buffer *uniform_buffer_memory = VulkanMemoryMap(uniform_region);
    if (uniform_buffer_memory == NULL) {
        return TEST_INVALID_PARAMETER;
    }
    uniform_buffer_memory->loop_count = loop_count;
    uniform_buffer_memory->texture_x_mask = texture_side - 1;
    uniform_buffer_memory->texture_pixel_mask = (texture_side * texture_side) - 1;
    uniform_buffer_memory->texture_side_shift = _VulkanRateTextureCachePowerOfTwoLog2(texture_side);
    uniform_buffer_memory->inv_texture_side = 1.0f / (float)texture_side;
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
        INFO("Loop count %lu workgroup count %llu texture side %lu took %.3fms (rate: N/A)\n", loop_count, workgroup_count, texture_side, time / 1000.0f);
        if (result != NULL) {
            *result = 0;
        }
    } else {
        uint64_t total_fetches = (uint64_t)groups_x * (uint64_t)groups_y * (uint64_t)groups_z * VULKAN_RATE_TEXTURE_CACHE_WORKGROUP_SIZE * VULKAN_RATE_TEXTURE_CACHE_PARALLEL_FETCHES * loop_count;
        uint64_t throughput_per_second = (total_fetches * 1000000) / time;
        helper_unit_pair unit_conversion;
        HelperConvertUnitsPlain1000(throughput_per_second, &unit_conversion);
        INFO("Loop count %lu workgroup count %llu texture side %lu took %.3fms (rate: %.3f %sTexel/s)\n", loop_count, workgroup_count, texture_side, time / 1000.0f, unit_conversion.value, unit_conversion.units);
        if (result != NULL) {
            *result = throughput_per_second;
        }
    }

cleanup_command_sequence:
    VulkanCommandBufferReset(command_sequence);
error:
    return status;
}

static uint32_t _VulkanRateTextureCachePowerOfTwoLog2(uint32_t value) {
    uint32_t shift = 0;
    while (value > 1) {
        value >>= 1;
        shift++;
    }
    return shift;
}

static const char *_VulkanRateTextureCacheGetShaderName(uint32_t operation) {
    switch (operation) {
    case VULKAN_RATE_TEXTURE_CACHE_OP_TEXEL_FETCH:
        return "vulkan_rate_texture_cache.spv";
    case VULKAN_RATE_TEXTURE_CACHE_OP_LINEAR:
        return "vulkan_rate_texture_cache_linear.spv";
    default:
        return NULL;
    }
}

static const char *_VulkanRateTextureCacheGetOperationName(uint32_t operation) {
    switch (operation) {
    case VULKAN_RATE_TEXTURE_CACHE_OP_TEXEL_FETCH:
        return "randomTexelFetch";
    case VULKAN_RATE_TEXTURE_CACHE_OP_LINEAR:
        return "randomLinear";
    default:
        return NULL;
    }
}
