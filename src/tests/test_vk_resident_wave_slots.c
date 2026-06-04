// SPDX-License-Identifier: MIT

#include "main.h"
#include "logger.h"
#include "vulkan_helper.h"
#include "vulkan_runner.h"
#include "vulkan_memory.h"
#include "vulkan_shader.h"
#include "vulkan_compute_pipeline.h"
#include "vulkan_command_buffer.h"
#include "tests/test_vk_resident_wave_slots.h"

#define VULKAN_RESIDENT_WAVE_SLOTS_BUFFER_WORDS       6u
#define VULKAN_RESIDENT_WAVE_SLOTS_SPIN_ITERATIONS    65536u
#define VULKAN_RESIDENT_WAVE_SLOTS_MAX_WORKGROUPS     4096u

typedef struct vulkan_resident_wave_slots_probe_t {
    uint32_t target_workgroups;
    uint32_t spin_iterations;
    uint32_t arrived;
    uint32_t passed;
    uint32_t timed_out;
    uint32_t sink;
} vulkan_resident_wave_slots_probe;

typedef struct vulkan_resident_wave_slots_result_t {
    uint32_t target_workgroups;
    uint32_t arrived;
    uint32_t passed;
    uint32_t timed_out;
} vulkan_resident_wave_slots_result;

static test_status _VulkanResidentWaveSlotsEntry(vulkan_physical_device *physical_device, void *config_data);
static test_status _VulkanResidentWaveSlotsRunProbe(vulkan_compute_pipeline *pipeline, vulkan_command_sequence *command_sequence, vulkan_region *probe_region, vulkan_region *readback_region, uint32_t target_workgroups, vulkan_resident_wave_slots_result *result);
static test_status _VulkanResidentWaveSlotsFindLimit(vulkan_compute_pipeline *pipeline, vulkan_command_sequence *command_sequence, vulkan_region *probe_region, vulkan_region *readback_region, uint32_t max_workgroups, uint32_t *resident_workgroups);
static bool _VulkanResidentWaveSlotsProbePassed(const vulkan_resident_wave_slots_result *result);

test_status TestsVulkanResidentWaveSlotsRegister() {
    return VulkanRunnerRegisterTest(&_VulkanResidentWaveSlotsEntry, NULL, TESTS_VULKAN_RESIDENT_WAVE_SLOTS_NAME, TESTS_VULKAN_RESIDENT_WAVE_SLOTS_VERSION, false);
}

static test_status _VulkanResidentWaveSlotsEntry(vulkan_physical_device *physical_device, void *config_data) {
    TEST_UNUSED(config_data);

    test_status status = TEST_OK;
    VkPhysicalDeviceSubgroupProperties subgroup_properties = {0};
    VkPhysicalDeviceProperties2 properties = {0};
    VkQueueFamilyProperties *queue_family_properties = NULL;
    uint32_t queue_family_count = 0;
    vulkan_device device = {0};
    vulkan_shader shader = {0};
    vulkan_memory probe_memory = {0};
    vulkan_memory readback_memory = {0};
    vulkan_command_buffer command_buffer = {0};
    vulkan_command_sequence command_sequence = {0};
    bool device_created = false;
    bool shader_initialized = false;
    bool probe_memory_initialized = false;
    bool probe_memory_allocated = false;
    bool readback_memory_initialized = false;
    bool readback_memory_allocated = false;
    bool command_buffer_initialized = false;
    bool command_sequence_initialized = false;

    subgroup_properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;
    properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    properties.pNext = &subgroup_properties;
    vkGetPhysicalDeviceProperties2(physical_device->physical_device, &properties);

    const uint32_t subgroup_size = subgroup_properties.subgroupSize;
    if (subgroup_size == 0) {
        WARNING("Could not query Vulkan subgroup size\n");
        return TEST_VK_FEATURE_UNSUPPORTED;
    }

    INFO("Device Name: %s\n", properties.properties.deviceName);
    INFO("Subgroup Size: %u\n", subgroup_size);

    status = VulkanGetPhysicalQueueFamilyProperties(physical_device, &queue_family_properties, &queue_family_count);
    TEST_RETFAIL(status);

    status = VulkanCreateDevice(physical_device, queue_family_properties, queue_family_count, VK_QUEUE_COMPUTE_BIT, &device, NULL);
    if (!TEST_SUCCESS(status)) {
        goto cleanup_queue_properties;
    }
    device_created = true;

    status = VulkanShaderInitializeFromFile(&device, "vulkan_resident_wave_slots.spv", VK_SHADER_STAGE_COMPUTE_BIT, &shader);
    if (!TEST_SUCCESS(status)) {
        goto cleanup_device;
    }
    shader_initialized = true;

    status = VulkanShaderAddDescriptor(&shader, "probe buffer", VULKAN_BINDING_STORAGE, 0, 0);
    if (!TEST_SUCCESS(status)) {
        goto cleanup_shader;
    }
    status = VulkanShaderCreateDescriptorSets(&shader);
    if (!TEST_SUCCESS(status)) {
        goto cleanup_shader;
    }

    status = VulkanMemoryInitialize(&device, VULKAN_MEMORY_NORMAL, &probe_memory);
    if (!TEST_SUCCESS(status)) {
        goto cleanup_shader;
    }
    probe_memory_initialized = true;
    status = VulkanMemoryAddRegion(&probe_memory, sizeof(vulkan_resident_wave_slots_probe), "probe buffer", VULKAN_REGION_STORAGE, VULKAN_REGION_TRANSFER_SOURCE | VULKAN_REGION_TRANSFER_DESTINATION);
    if (!TEST_SUCCESS(status)) {
        goto cleanup_probe_memory;
    }
    status = VulkanMemoryAllocateBacking(&probe_memory);
    if (!TEST_SUCCESS(status)) {
        goto cleanup_probe_memory;
    }
    probe_memory_allocated = true;

    status = VulkanMemoryInitialize(&device, VULKAN_MEMORY_VISIBLE | VULKAN_MEMORY_HOST_LOCAL, &readback_memory);
    if (!TEST_SUCCESS(status)) {
        goto cleanup_probe_memory_backing;
    }
    readback_memory_initialized = true;
    status = VulkanMemoryAddRegion(&readback_memory, sizeof(vulkan_resident_wave_slots_probe), "probe readback", VULKAN_REGION_STORAGE, VULKAN_REGION_TRANSFER_DESTINATION);
    if (!TEST_SUCCESS(status)) {
        goto cleanup_readback_memory;
    }
    status = VulkanMemoryAllocateBacking(&readback_memory);
    if (!TEST_SUCCESS(status)) {
        goto cleanup_readback_memory;
    }
    readback_memory_allocated = true;

    vulkan_region *probe_region = VulkanMemoryGetRegion(&probe_memory, "probe buffer");
    vulkan_region *readback_region = VulkanMemoryGetRegion(&readback_memory, "probe readback");
    if (probe_region == NULL) {
        status = TEST_VK_BINDING_UNKNOWN_MEMORY_REGION;
        goto cleanup_readback_memory_backing;
    }
    if (readback_region == NULL) {
        status = TEST_VK_BINDING_UNKNOWN_MEMORY_REGION;
        goto cleanup_readback_memory_backing;
    }

    status = VulkanCommandBufferInitialize(&device, 1, &command_buffer);
    if (!TEST_SUCCESS(status)) {
        goto cleanup_readback_memory_backing;
    }
    command_buffer_initialized = true;

    status = VulkanCommandSequenceInitialize(&command_buffer, VULKAN_COMMAND_SEQUENCE_RESET_ON_COMPLETION, &command_sequence);
    if (!TEST_SUCCESS(status)) {
        goto cleanup_command_buffer;
    }
    command_sequence_initialized = true;

    const uint32_t max_invocations = properties.properties.limits.maxComputeWorkGroupInvocations;
    const uint32_t max_local_size_x = properties.properties.limits.maxComputeWorkGroupSize[0];
    uint32_t max_waves_per_workgroup = max_invocations / subgroup_size;
    uint32_t max_waves_by_x = max_local_size_x / subgroup_size;
    if (max_waves_by_x < max_waves_per_workgroup) {
        max_waves_per_workgroup = max_waves_by_x;
    }
    if (max_waves_per_workgroup == 0) {
        WARNING("Subgroup size %u exceeds max compute workgroup size\n", subgroup_size);
        status = TEST_VK_FEATURE_UNSUPPORTED;
        goto cleanup_command_sequence;
    }

    uint32_t max_probe_workgroups = properties.properties.limits.maxComputeWorkGroupCount[0];
    if (max_probe_workgroups > VULKAN_RESIDENT_WAVE_SLOTS_MAX_WORKGROUPS) {
        max_probe_workgroups = VULKAN_RESIDENT_WAVE_SLOTS_MAX_WORKGROUPS;
    }

    if (MainGetTestResultFormat() == test_result_csv) {
        LOG_PLAIN("device_index,device_name,subgroup_size,waves_per_workgroup,local_size_x,resident_workgroups,total_resident_waves\n");
    }

    uint32_t max_resident_waves = 0;
    uint32_t waves_per_workgroup = 1;
    while (waves_per_workgroup <= max_waves_per_workgroup) {
        const uint32_t local_size_x = subgroup_size * waves_per_workgroup;
        VkSpecializationMapEntry specialization_entry = {0};
        VkSpecializationInfo specialization_info = {0};
        vulkan_compute_pipeline pipeline = {0};
        uint32_t resident_workgroups = 0;

        specialization_entry.constantID = 0;
        specialization_entry.offset = 0;
        specialization_entry.size = sizeof(local_size_x);
        specialization_info.mapEntryCount = 1;
        specialization_info.pMapEntries = &specialization_entry;
        specialization_info.dataSize = sizeof(local_size_x);
        specialization_info.pData = &local_size_x;

        status = VulkanComputePipelineInitializeSpecialized(&shader, "main", &specialization_info, &pipeline);
        if (!TEST_SUCCESS(status)) {
            goto cleanup_command_sequence;
        }
        status = VulkanComputePipelineBind(&pipeline, &probe_memory, "probe buffer");
        if (!TEST_SUCCESS(status)) {
            VulkanComputePipelineCleanUp(&pipeline);
            goto cleanup_command_sequence;
        }

        status = _VulkanResidentWaveSlotsFindLimit(&pipeline, &command_sequence, probe_region, readback_region, max_probe_workgroups, &resident_workgroups);
        VulkanComputePipelineCleanUp(&pipeline);
        if (!TEST_SUCCESS(status)) {
            goto cleanup_command_sequence;
        }

        const uint32_t resident_waves = resident_workgroups * waves_per_workgroup;
        if (resident_waves > max_resident_waves) {
            max_resident_waves = resident_waves;
        }
        if (MainGetTestResultFormat() == test_result_csv) {
            LOG_PLAIN("%u,\"%s\",%u,%u,%u,%u,%u\n",
                      physical_device->device_index,
                      properties.properties.deviceName,
                      subgroup_size,
                      waves_per_workgroup,
                      local_size_x,
                      resident_workgroups,
                      resident_waves);
        } else if (MainGetTestResultFormat() == test_result_raw) {
            LOG_RESULT(waves_per_workgroup, "%u", "%u", waves_per_workgroup, resident_waves);
        } else {
            INFO("waves/workgroup=%u local_size_x=%u resident_workgroups=%u total_resident_waves=%u\n",
                 waves_per_workgroup,
                 local_size_x,
                 resident_workgroups,
                 resident_waves);
        }

        if (waves_per_workgroup == max_waves_per_workgroup) {
            break;
        }
        if (waves_per_workgroup > max_waves_per_workgroup / 2) {
            waves_per_workgroup = max_waves_per_workgroup;
        } else {
            waves_per_workgroup *= 2;
        }
    }
    INFO("Max total resident waves observed: %u\n", max_resident_waves);
    INFO("Divide this by known CU/SM count to estimate resident waves per CU/SM\n");

cleanup_command_sequence:
    if (command_sequence_initialized) {
        VulkanCommandBufferReset(&command_sequence);
    }
cleanup_command_buffer:
    if (command_buffer_initialized) {
        VulkanCommandBufferCleanUp(&command_buffer);
    }
cleanup_readback_memory_backing:
    if (readback_memory_allocated) {
        VulkanMemoryFreeBuffersAndBacking(&readback_memory);
    }
cleanup_readback_memory:
    if (readback_memory_initialized) {
        VulkanMemoryCleanUp(&readback_memory);
    }
cleanup_probe_memory_backing:
    if (probe_memory_allocated) {
        VulkanMemoryFreeBuffersAndBacking(&probe_memory);
    }
cleanup_probe_memory:
    if (probe_memory_initialized) {
        VulkanMemoryCleanUp(&probe_memory);
    }
cleanup_shader:
    if (shader_initialized) {
        VulkanShaderCleanUp(&shader);
    }
cleanup_device:
    if (device_created) {
        VulkanDestroyDevice(&device);
    }
cleanup_queue_properties:
    free(queue_family_properties);
    return status;
}

static test_status _VulkanResidentWaveSlotsRunProbe(vulkan_compute_pipeline *pipeline, vulkan_command_sequence *command_sequence, vulkan_region *probe_region, vulkan_region *readback_region, uint32_t target_workgroups, vulkan_resident_wave_slots_result *result) {
    vulkan_resident_wave_slots_probe initial_probe = {0};
    initial_probe.target_workgroups = target_workgroups;
    initial_probe.spin_iterations = VULKAN_RESIDENT_WAVE_SLOTS_SPIN_ITERATIONS;

    test_status status = VulkanCommandBufferStart(command_sequence);
    TEST_RETFAIL(status);

    VkBufferMemoryBarrier probe_transfer_to_shader = {0};
    probe_transfer_to_shader.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    probe_transfer_to_shader.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    probe_transfer_to_shader.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    probe_transfer_to_shader.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    probe_transfer_to_shader.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    probe_transfer_to_shader.buffer = probe_region->backing_buffer;
    probe_transfer_to_shader.offset = probe_region->offset;
    probe_transfer_to_shader.size = probe_region->size;

    vkCmdUpdateBuffer(command_sequence->command_buffer, probe_region->backing_buffer, probe_region->offset, sizeof(initial_probe), &initial_probe);
    vkCmdPipelineBarrier(command_sequence->command_buffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, NULL, 1, &probe_transfer_to_shader, 0, NULL);

    status = VulkanCommandBufferBindComputePipeline(command_sequence, pipeline);
    if (!TEST_SUCCESS(status)) {
        goto reset_command_sequence;
    }
    status = VulkanCommandBufferDispatch(command_sequence, target_workgroups, 1, 1);
    if (!TEST_SUCCESS(status)) {
        goto reset_command_sequence;
    }

    VkBufferMemoryBarrier probe_shader_to_transfer = {0};
    probe_shader_to_transfer.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    probe_shader_to_transfer.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    probe_shader_to_transfer.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    probe_shader_to_transfer.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    probe_shader_to_transfer.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    probe_shader_to_transfer.buffer = probe_region->backing_buffer;
    probe_shader_to_transfer.offset = probe_region->offset;
    probe_shader_to_transfer.size = probe_region->size;
    vkCmdPipelineBarrier(command_sequence->command_buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, NULL, 1, &probe_shader_to_transfer, 0, NULL);

    status = VulkanCommandBufferCopyRegion(command_sequence, probe_region, readback_region);
    if (!TEST_SUCCESS(status)) {
        goto reset_command_sequence;
    }

    VkBufferMemoryBarrier readback_transfer_to_host = {0};
    readback_transfer_to_host.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    readback_transfer_to_host.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    readback_transfer_to_host.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    readback_transfer_to_host.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    readback_transfer_to_host.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    readback_transfer_to_host.buffer = readback_region->backing_buffer;
    readback_transfer_to_host.offset = readback_region->offset;
    readback_transfer_to_host.size = readback_region->size;
    vkCmdPipelineBarrier(command_sequence->command_buffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 0, NULL, 1, &readback_transfer_to_host, 0, NULL);

    status = VulkanCommandBufferEnd(command_sequence);
    if (!TEST_SUCCESS(status)) {
        goto reset_command_sequence;
    }
    status = VulkanCommandBufferSubmit(command_sequence);
    if (!TEST_SUCCESS(status)) {
        goto reset_command_sequence;
    }
    status = VulkanCommandBufferWait(command_sequence, VULKAN_COMMAND_SEQUENCE_WAIT_INFINITE);
    if (!TEST_SUCCESS(status)) {
        goto reset_command_sequence;
    }

    vulkan_resident_wave_slots_probe *probe = (vulkan_resident_wave_slots_probe *)VulkanMemoryMap(readback_region);
    if (probe == NULL) {
        return TEST_VK_MEMORY_MAPPING_ERROR;
    }
    result->target_workgroups = target_workgroups;
    result->arrived = probe->arrived;
    result->passed = probe->passed;
    result->timed_out = probe->timed_out;
    VulkanMemoryUnmap(readback_region);
    return TEST_OK;

reset_command_sequence:
    VulkanCommandBufferReset(command_sequence);
    return status;
}

static test_status _VulkanResidentWaveSlotsFindLimit(vulkan_compute_pipeline *pipeline, vulkan_command_sequence *command_sequence, vulkan_region *probe_region, vulkan_region *readback_region, uint32_t max_workgroups, uint32_t *resident_workgroups) {
    test_status status = TEST_OK;
    vulkan_resident_wave_slots_result result = {0};
    uint32_t low = 0;
    uint32_t high = 1;

    while (high <= max_workgroups) {
        status = _VulkanResidentWaveSlotsRunProbe(pipeline, command_sequence, probe_region, readback_region, high, &result);
        TEST_RETFAIL(status);
        if (!_VulkanResidentWaveSlotsProbePassed(&result)) {
            break;
        }
        low = high;
        if (high > max_workgroups / 2) {
            high = max_workgroups + 1;
            break;
        }
        high *= 2;
    }

    if (high > max_workgroups) {
        high = max_workgroups + 1;
    }

    while (low + 1 < high) {
        uint32_t mid = low + ((high - low) / 2);
        status = _VulkanResidentWaveSlotsRunProbe(pipeline, command_sequence, probe_region, readback_region, mid, &result);
        TEST_RETFAIL(status);
        if (_VulkanResidentWaveSlotsProbePassed(&result)) {
            low = mid;
        } else {
            high = mid;
        }
    }

    *resident_workgroups = low;
    return TEST_OK;
}

static bool _VulkanResidentWaveSlotsProbePassed(const vulkan_resident_wave_slots_result *result) {
    return result->timed_out == 0 && result->arrived == result->target_workgroups && result->passed == result->target_workgroups;
}
