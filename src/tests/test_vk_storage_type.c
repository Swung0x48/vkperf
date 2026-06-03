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
#include "buffer_filler.h"
#include "tests/test_vk_storage_type.h"

#define VULKAN_STORAGE_TYPE_WIDTH                 (1024)
#define VULKAN_STORAGE_TYPE_HEIGHT                (1024)
#define VULKAN_STORAGE_TYPE_BUFFER_WORKGROUP      (256)
#define VULKAN_STORAGE_TYPE_TEXTURE_WORKGROUP_DIM (16)
#define VULKAN_STORAGE_TYPE_WARMUP_FRAMES         (5)
#define VULKAN_STORAGE_TYPE_BENCHMARK_FRAMES      (10)
#define VULKAN_STORAGE_TYPE_RNG_SEED              (332487265)

#define VULKAN_STORAGE_TYPE_PATTERN_UNIFORM       (0)
#define VULKAN_STORAGE_TYPE_PATTERN_LINEAR        (1)
#define VULKAN_STORAGE_TYPE_PATTERN_RANDOM        (2)

typedef struct vulkan_storage_type_uniform_buffer_t {
    uint32_t elements_mask;
    uint32_t write_index;
    uint32_t read_start_address;
    uint32_t access_pattern;
} vulkan_storage_type_uniform_buffer;

typedef struct vulkan_storage_type_case_t {
    const char *name;
    bool use_texture;
    uint32_t access_pattern;
} vulkan_storage_type_case;

static const vulkan_storage_type_case vulkan_storage_type_cases[] = {
    { "Buffer<RGBA8>.Load uniform", false, VULKAN_STORAGE_TYPE_PATTERN_UNIFORM },
    { "Buffer<RGBA8>.Load linear", false, VULKAN_STORAGE_TYPE_PATTERN_LINEAR },
    { "Buffer<RGBA8>.Load random", false, VULKAN_STORAGE_TYPE_PATTERN_RANDOM },
    { "Texture2D<RGBA8>.Load uniform", true, VULKAN_STORAGE_TYPE_PATTERN_UNIFORM },
    { "Texture2D<RGBA8>.Load linear", true, VULKAN_STORAGE_TYPE_PATTERN_LINEAR },
    { "Texture2D<RGBA8>.Load random", true, VULKAN_STORAGE_TYPE_PATTERN_RANDOM },
};

static test_status _VulkanStorageTypeEntry(vulkan_physical_device *device, void *config_data);
static test_status _VulkanStorageTypeFillTexture(vulkan_texture *texture);
static test_status _VulkanStorageTypeRunCase(vulkan_compute_pipeline *pipeline, vulkan_command_sequence *command_sequence, vulkan_region *uniform_region, const vulkan_storage_type_case *test_case, uint64_t *total_time_us);

test_status TestsVulkanStorageTypeRegister() {
    return VulkanRunnerRegisterTest(&_VulkanStorageTypeEntry, NULL, TESTS_VULKAN_STORAGE_TYPE_NAME, TESTS_VULKAN_STORAGE_TYPE_VERSION, false);
}

static test_status _VulkanStorageTypeEntry(vulkan_physical_device *physical_device, void *config_data) {
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

    uint32_t *group_limits = device.physical_device->physical_properties.properties.limits.maxComputeWorkGroupCount;
    uint32_t buffer_groups = (VULKAN_STORAGE_TYPE_WIDTH * VULKAN_STORAGE_TYPE_HEIGHT) / VULKAN_STORAGE_TYPE_BUFFER_WORKGROUP;
    if (buffer_groups > group_limits[0] || 64 > group_limits[0] || 64 > group_limits[1]) {
        status = TEST_VK_DISPATCH_TOO_LARGE;
        goto cleanup_device;
    }

    vulkan_shader buffer_shader;
    status = VulkanShaderInitializeFromFile(&device, "vulkan_storage_type_buffer.spv", VK_SHADER_STAGE_COMPUTE_BIT, &buffer_shader);
    if (!TEST_SUCCESS(status)) {
        goto cleanup_device;
    }
    status = VulkanShaderAddDescriptor(&buffer_shader, "source buffer", VULKAN_BINDING_STORAGE, 0, 0);
    if (!TEST_SUCCESS(status)) {
        goto cleanup_buffer_shader;
    }
    status = VulkanShaderAddDescriptor(&buffer_shader, "output buffer", VULKAN_BINDING_STORAGE, 0, 1);
    if (!TEST_SUCCESS(status)) {
        goto cleanup_buffer_shader;
    }
    status = VulkanShaderAddFixedSizeDescriptor(&buffer_shader, sizeof(vulkan_storage_type_uniform_buffer), "uniform buffer", VULKAN_BINDING_UNIFORM, 0, 2);
    if (!TEST_SUCCESS(status)) {
        goto cleanup_buffer_shader;
    }
    status = VulkanShaderCreateDescriptorSets(&buffer_shader);
    if (!TEST_SUCCESS(status)) {
        goto cleanup_buffer_shader;
    }
    vulkan_compute_pipeline buffer_pipeline;
    status = VulkanComputePipelineInitialize(&buffer_shader, "main", &buffer_pipeline);
    if (!TEST_SUCCESS(status)) {
        goto cleanup_buffer_shader;
    }

    vulkan_shader texture_shader;
    status = VulkanShaderInitializeFromFile(&device, "vulkan_storage_type_texture.spv", VK_SHADER_STAGE_COMPUTE_BIT, &texture_shader);
    if (!TEST_SUCCESS(status)) {
        goto cleanup_buffer_pipeline;
    }
    status = VulkanShaderAddDescriptor(&texture_shader, "source texture", VULKAN_BINDING_SAMPLER, 0, 0);
    if (!TEST_SUCCESS(status)) {
        goto cleanup_texture_shader;
    }
    status = VulkanShaderAddDescriptor(&texture_shader, "output buffer", VULKAN_BINDING_STORAGE, 0, 1);
    if (!TEST_SUCCESS(status)) {
        goto cleanup_texture_shader;
    }
    status = VulkanShaderAddFixedSizeDescriptor(&texture_shader, sizeof(vulkan_storage_type_uniform_buffer), "uniform buffer", VULKAN_BINDING_UNIFORM, 0, 2);
    if (!TEST_SUCCESS(status)) {
        goto cleanup_texture_shader;
    }
    status = VulkanShaderCreateDescriptorSets(&texture_shader);
    if (!TEST_SUCCESS(status)) {
        goto cleanup_texture_shader;
    }
    vulkan_compute_pipeline texture_pipeline;
    status = VulkanComputePipelineInitialize(&texture_shader, "main", &texture_pipeline);
    if (!TEST_SUCCESS(status)) {
        goto cleanup_texture_shader;
    }

    vulkan_memory memory;
    status = VulkanMemoryInitialize(&device, VULKAN_MEMORY_NORMAL, &memory);
    if (!TEST_SUCCESS(status)) {
        goto cleanup_texture_pipeline;
    }

    size_t texel_count = VULKAN_STORAGE_TYPE_WIDTH * VULKAN_STORAGE_TYPE_HEIGHT;
    size_t rgba8_size = texel_count * sizeof(uint32_t);
    status = VulkanMemoryAddRegion(&memory, rgba8_size, "source buffer", VULKAN_REGION_STORAGE, VULKAN_REGION_TRANSFER_DESTINATION);
    if (!TEST_SUCCESS(status)) {
        goto cleanup_memory;
    }
    status = VulkanMemoryAddTexture2D(&memory, VULKAN_STORAGE_TYPE_WIDTH, VULKAN_STORAGE_TYPE_HEIGHT, VK_FORMAT_R8G8B8A8_UNORM, 1, "source texture");
    if (!TEST_SUCCESS(status)) {
        goto cleanup_memory;
    }
    status = VulkanMemoryAddRegion(&memory, rgba8_size, "output buffer", VULKAN_REGION_STORAGE, VULKAN_REGION_NORMAL);
    if (!TEST_SUCCESS(status)) {
        goto cleanup_memory;
    }
    status = VulkanMemoryAllocateBacking(&memory);
    if (!TEST_SUCCESS(status)) {
        goto cleanup_memory;
    }

    vulkan_region *source_buffer = VulkanMemoryGetRegion(&memory, "source buffer");
    status = BufferFillerRandomIntegers(source_buffer, VULKAN_STORAGE_TYPE_RNG_SEED);
    if (!TEST_SUCCESS(status)) {
        goto free_memory;
    }

    vulkan_texture *source_texture = VulkanMemoryGetTexture(&memory, "source texture");
    status = _VulkanStorageTypeFillTexture(source_texture);
    if (!TEST_SUCCESS(status)) {
        goto free_memory;
    }

    vulkan_memory uniform_memory;
    status = VulkanMemoryInitialize(&device, VULKAN_MEMORY_VISIBLE | VULKAN_MEMORY_HOST_LOCAL, &uniform_memory);
    if (!TEST_SUCCESS(status)) {
        goto free_memory;
    }
    status = VulkanMemoryAddRegion(&uniform_memory, sizeof(vulkan_storage_type_uniform_buffer), "uniform buffer", VULKAN_REGION_UNIFORM, VULKAN_REGION_NORMAL);
    if (!TEST_SUCCESS(status)) {
        goto cleanup_uniform_memory;
    }
    status = VulkanMemoryAllocateBacking(&uniform_memory);
    if (!TEST_SUCCESS(status)) {
        goto cleanup_uniform_memory;
    }

    status = VulkanComputePipelineBind(&buffer_pipeline, &memory, "source buffer");
    if (!TEST_SUCCESS(status)) {
        goto free_uniform_memory;
    }
    status = VulkanComputePipelineBind(&buffer_pipeline, &memory, "output buffer");
    if (!TEST_SUCCESS(status)) {
        goto free_uniform_memory;
    }
    status = VulkanComputePipelineBind(&buffer_pipeline, &uniform_memory, "uniform buffer");
    if (!TEST_SUCCESS(status)) {
        goto free_uniform_memory;
    }
    status = VulkanComputePipelineBind(&texture_pipeline, &memory, "source texture");
    if (!TEST_SUCCESS(status)) {
        goto free_uniform_memory;
    }
    status = VulkanComputePipelineBind(&texture_pipeline, &memory, "output buffer");
    if (!TEST_SUCCESS(status)) {
        goto free_uniform_memory;
    }
    status = VulkanComputePipelineBind(&texture_pipeline, &uniform_memory, "uniform buffer");
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

    uint32_t case_count = (uint32_t)(sizeof(vulkan_storage_type_cases) / sizeof(vulkan_storage_type_cases[0]));
    uint64_t *results = malloc(case_count * sizeof(uint64_t));
    if (results == NULL) {
        status = TEST_OUT_OF_MEMORY;
        goto cleanup_command_buffer;
    }

    vulkan_region *uniform_region = VulkanMemoryGetRegion(&uniform_memory, "uniform buffer");
    for (uint32_t i = 0; i < case_count; i++) {
        vulkan_compute_pipeline *pipeline = vulkan_storage_type_cases[i].use_texture ? &texture_pipeline : &buffer_pipeline;
        INFO("Running %s...\n", vulkan_storage_type_cases[i].name);
        status = _VulkanStorageTypeRunCase(pipeline, &command_sequence, uniform_region, &(vulkan_storage_type_cases[i]), &(results[i]));
        if (!TEST_SUCCESS(status)) {
            goto free_results;
        }
    }

    uint64_t baseline = results[2];
    INFO("Final results for %s:\n", physical_device->physical_properties.properties.deviceName);
    if (MainGetTestResultFormat() == test_result_csv) {
        LOG_PLAIN("%s,\n", physical_device->physical_properties.properties.deviceName);
        LOG_PLAIN("Test,Total Time (ms),Relative to Buffer<RGBA8>.Load random\n");
    }
    for (uint32_t i = 0; i < case_count; i++) {
        double total_ms = (double)results[i] / 1000.0;
        double relative = results[i] == 0 ? 0.0 : (double)baseline / (double)results[i];
        if (MainGetTestResultFormat() == test_result_csv) {
            LOG_PLAIN("%s,%.3f,%.3f\n", vulkan_storage_type_cases[i].name, total_ms, relative);
        } else if (MainGetTestResultFormat() == test_result_raw) {
            LOG_RESULT(i, "%llu", "%llu", (uint64_t)(relative * 1000000.0), results[i]);
        } else {
            INFO("%s: %.3fms (relative %.3fx)\n", vulkan_storage_type_cases[i].name, total_ms, relative);
        }
    }

free_results:
    free(results);
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
cleanup_texture_pipeline:
    VulkanComputePipelineCleanUp(&texture_pipeline);
cleanup_texture_shader:
    VulkanShaderCleanUp(&texture_shader);
cleanup_buffer_pipeline:
    VulkanComputePipelineCleanUp(&buffer_pipeline);
cleanup_buffer_shader:
    VulkanShaderCleanUp(&buffer_shader);
cleanup_device:
    VulkanDestroyDevice(&device);
cleanup_queue_properties:
    free(queue_family_properties);
error:
    return status;
}

static test_status _VulkanStorageTypeFillTexture(vulkan_texture *texture) {
    test_status status = VulkanTexturePrepareForCopy(texture);
    TEST_RETFAIL(status);

    size_t texture_size = VULKAN_STORAGE_TYPE_WIDTH * VULKAN_STORAGE_TYPE_HEIGHT * sizeof(uint32_t);
    vulkan_staging staging;
    status = VulkanStagingInitializeImage(texture, texture_size, &staging);
    TEST_RETFAIL(status);

    uint32_t *staging_data = (uint32_t *)VulkanStagingGetBuffer(&staging);
    if (staging_data == NULL) {
        VulkanStagingCleanUp(&staging);
        return TEST_VK_MEMORY_MAPPING_ERROR;
    }
    uint64_t rng_state;
    HelperSeedRandom(&rng_state, VULKAN_STORAGE_TYPE_RNG_SEED);
    for (size_t i = 0; i < VULKAN_STORAGE_TYPE_WIDTH * VULKAN_STORAGE_TYPE_HEIGHT; i++) {
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

static test_status _VulkanStorageTypeRunCase(vulkan_compute_pipeline *pipeline, vulkan_command_sequence *command_sequence, vulkan_region *uniform_region, const vulkan_storage_type_case *test_case, uint64_t *total_time_us) {
    test_status status = TEST_OK;
    volatile vulkan_storage_type_uniform_buffer *uniform_buffer = VulkanMemoryMap(uniform_region);
    if (uniform_buffer == NULL) {
        return TEST_VK_MEMORY_MAPPING_ERROR;
    }
    uniform_buffer->elements_mask = 0;
    uniform_buffer->write_index = 0xffffffffu;
    uniform_buffer->read_start_address = 0;
    uniform_buffer->access_pattern = test_case->access_pattern;
    VulkanMemoryUnmap(uniform_region);

    uint32_t dispatch_x = test_case->use_texture ? (VULKAN_STORAGE_TYPE_WIDTH / VULKAN_STORAGE_TYPE_TEXTURE_WORKGROUP_DIM) : ((VULKAN_STORAGE_TYPE_WIDTH * VULKAN_STORAGE_TYPE_HEIGHT) / VULKAN_STORAGE_TYPE_BUFFER_WORKGROUP);
    uint32_t dispatch_y = test_case->use_texture ? (VULKAN_STORAGE_TYPE_HEIGHT / VULKAN_STORAGE_TYPE_TEXTURE_WORKGROUP_DIM) : 1;
    *total_time_us = 0;

    for (uint32_t i = 0; i < (VULKAN_STORAGE_TYPE_WARMUP_FRAMES + VULKAN_STORAGE_TYPE_BENCHMARK_FRAMES); i++) {
        status = VulkanCommandBufferStart(command_sequence);
        TEST_RETFAIL(status);
        status = VulkanCommandBufferBindComputePipeline(command_sequence, pipeline);
        if (!TEST_SUCCESS(status)) {
            goto reset_command_sequence;
        }
        status = VulkanCommandBufferDispatch(command_sequence, dispatch_x, dispatch_y, 1);
        if (!TEST_SUCCESS(status)) {
            goto reset_command_sequence;
        }
        status = VulkanCommandBufferEnd(command_sequence);
        if (!TEST_SUCCESS(status)) {
            goto reset_command_sequence;
        }
        HelperResetTimestamp();
        status = VulkanCommandBufferSubmit(command_sequence);
        if (!TEST_SUCCESS(status)) {
            goto reset_command_sequence;
        }
        status = VulkanCommandBufferWait(command_sequence, VULKAN_COMMAND_SEQUENCE_WAIT_INFINITE);
        if (!TEST_SUCCESS(status)) {
            goto reset_command_sequence;
        }
        uint64_t elapsed = HelperMarkTimestamp();
        if (i >= VULKAN_STORAGE_TYPE_WARMUP_FRAMES) {
            *total_time_us += elapsed;
        }
    }
    return TEST_OK;

reset_command_sequence:
    VulkanCommandBufferReset(command_sequence);
    return status;
}
