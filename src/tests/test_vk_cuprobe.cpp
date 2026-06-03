// SPDX-License-Identifier: MIT

#include "main.h"
#include "logger.h"
#include "vulkan_helper.h"
#include "vulkan_runner.h"
#include "tests/test_vk_cuprobe.h"
#include "tests/test_vk_cuprobe_spv.h"

#include <cstring>
#include <vector>

#define VULKAN_CUPROBE_BUFFER_FLOATS             (1024u * 256u)
#define VULKAN_CUPROBE_WORKGROUP_COUNT_MAX       (512u)

typedef struct vulkan_cuprobe_buffer_t {
    VkBuffer buffer;
    VkDeviceMemory memory;
} vulkan_cuprobe_buffer;

static test_status _VulkanCUProbeEntry(vulkan_physical_device *physical_device, void *config_data);
static test_status _VulkanCUProbeCreateBuffer(vulkan_device *device, VkDeviceSize size, vulkan_cuprobe_buffer *buffer);
static void _VulkanCUProbeDestroyBuffer(vulkan_device *device, vulkan_cuprobe_buffer *buffer);
static test_status _VulkanCUProbeFindMemoryType(vulkan_physical_device *physical_device, uint32_t type_filter, VkMemoryPropertyFlags properties, uint32_t *memory_type_index);

static test_status _VulkanCUProbeEntry(vulkan_physical_device *physical_device, void *config_data) {
    TEST_UNUSED(config_data);

    test_status status = TEST_OK;
    VkResult res = VK_SUCCESS;
    VkQueueFamilyProperties *queue_family_properties = NULL;
    uint32_t queue_family_count = 0;
    vulkan_device device = {};
    vulkan_cuprobe_buffer input_buffer = {};
    vulkan_cuprobe_buffer output_buffer = {};
    VkQueryPool query_pool = VK_NULL_HANDLE;
    VkCommandPool command_pool = VK_NULL_HANDLE;
    VkDescriptorSetLayout descriptor_set_layout = VK_NULL_HANDLE;
    VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
    VkShaderModule shader_module = VK_NULL_HANDLE;
    VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkCommandBuffer command_buffer = VK_NULL_HANDLE;
    VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    uint32_t compute_queue_index = 0;
    const VkDeviceSize buffer_size = (VkDeviceSize)VULKAN_CUPROBE_BUFFER_FLOATS * sizeof(float);
    void *mapped = NULL;
    VkPhysicalDeviceProperties properties = {};
    uint32_t shared_memory_size = 0;
    std::vector<uint32_t> shader_words((__comp_spv_len + sizeof(uint32_t) - 1) / sizeof(uint32_t));
    std::vector<double> timings;
    timings.reserve(VULKAN_CUPROBE_WORKGROUP_COUNT_MAX);
    VkQueryPoolCreateInfo query_pool_info = {};
    VkCommandPoolCreateInfo command_pool_info = {};
    VkDescriptorSetLayoutBinding bindings[2] = {};
    VkDescriptorSetLayoutCreateInfo descriptor_layout_info = {};
    VkDescriptorPoolSize descriptor_pool_size = {};
    VkDescriptorPoolCreateInfo descriptor_pool_info = {};
    VkDescriptorSetAllocateInfo descriptor_set_alloc_info = {};
    VkDescriptorBufferInfo input_buffer_info = {};
    VkDescriptorBufferInfo output_buffer_info = {};
    VkWriteDescriptorSet descriptor_writes[2] = {};
    VkShaderModuleCreateInfo shader_module_info = {};
    VkSpecializationMapEntry specialization_entry = {};
    VkSpecializationInfo specialization_info = {};
    VkPipelineLayoutCreateInfo pipeline_layout_info = {};
    VkComputePipelineCreateInfo pipeline_info = {};
    VkCommandBufferAllocateInfo command_buffer_alloc_info = {};

    status = VulkanGetPhysicalQueueFamilyProperties(physical_device, &queue_family_properties, &queue_family_count);
    TEST_RETFAIL(status);

    status = VulkanSelectQueueFamily(&compute_queue_index, queue_family_properties, queue_family_count, VK_QUEUE_COMPUTE_BIT);
    if (!TEST_SUCCESS(status)) {
        goto cleanup_queue_properties;
    }

    if (queue_family_properties[compute_queue_index].timestampValidBits == 0) {
        WARNING("Timestamp queries are not supported by the selected compute queue\n");
        status = TEST_VK_FEATURE_UNSUPPORTED;
        goto cleanup_queue_properties;
    }

    status = VulkanCreateDeviceWithQueue(physical_device, queue_family_properties, compute_queue_index, 1, &device, NULL);
    if (!TEST_SUCCESS(status)) {
        goto cleanup_queue_properties;
    }
    queue = device.queues[0];

    query_pool_info.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
    query_pool_info.queryType = VK_QUERY_TYPE_TIMESTAMP;
    query_pool_info.queryCount = 2;
    res = vkCreateQueryPool(device.device, &query_pool_info, NULL, &query_pool);
    if (!VULKAN_SUCCESS(res)) {
        WARNING("Failed to create query pool: %li\n", res);
        status = TEST_UNKNOWN_ERROR;
        goto cleanup_device;
    }

    command_pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    command_pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    command_pool_info.queueFamilyIndex = compute_queue_index;
    res = vkCreateCommandPool(device.device, &command_pool_info, NULL, &command_pool);
    if (!VULKAN_SUCCESS(res)) {
        WARNING("Failed to create command pool: %li\n", res);
        status = TEST_VK_COMMAND_POOL_CREATION_ERROR;
        goto cleanup_query_pool;
    }

    status = _VulkanCUProbeCreateBuffer(&device, buffer_size, &input_buffer);
    if (!TEST_SUCCESS(status)) {
        goto cleanup_command_pool;
    }
    status = _VulkanCUProbeCreateBuffer(&device, buffer_size, &output_buffer);
    if (!TEST_SUCCESS(status)) {
        goto cleanup_input_buffer;
    }

    res = vkMapMemory(device.device, input_buffer.memory, 0, buffer_size, 0, &mapped);
    if (VULKAN_SUCCESS(res)) {
        float *input_data = (float *)mapped;
        for (uint32_t i = 0; i < VULKAN_CUPROBE_BUFFER_FLOATS; i++) {
            input_data[i] = 1.0f;
        }
        vkUnmapMemory(device.device, input_buffer.memory);
    }

    bindings[0].binding = 0;
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    bindings[1].binding = 1;
    bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[1].descriptorCount = 1;
    bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    descriptor_layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    descriptor_layout_info.bindingCount = 2;
    descriptor_layout_info.pBindings = bindings;
    res = vkCreateDescriptorSetLayout(device.device, &descriptor_layout_info, NULL, &descriptor_set_layout);
    if (!VULKAN_SUCCESS(res)) {
        WARNING("Failed to create descriptor set layout: %li\n", res);
        status = TEST_VK_DESCRIPTOR_SET_LAYOUT_CREATION_ERROR;
        goto cleanup_output_buffer;
    }

    descriptor_pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptor_pool_size.descriptorCount = 2;
    descriptor_pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    descriptor_pool_info.poolSizeCount = 1;
    descriptor_pool_info.pPoolSizes = &descriptor_pool_size;
    descriptor_pool_info.maxSets = 1;
    res = vkCreateDescriptorPool(device.device, &descriptor_pool_info, NULL, &descriptor_pool);
    if (!VULKAN_SUCCESS(res)) {
        WARNING("Failed to create descriptor pool: %li\n", res);
        status = TEST_VK_DESCRIPTOR_POOL_CREATION_ERROR;
        goto cleanup_descriptor_set_layout;
    }

    descriptor_set_alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    descriptor_set_alloc_info.descriptorPool = descriptor_pool;
    descriptor_set_alloc_info.descriptorSetCount = 1;
    descriptor_set_alloc_info.pSetLayouts = &descriptor_set_layout;
    res = vkAllocateDescriptorSets(device.device, &descriptor_set_alloc_info, &descriptor_set);
    if (!VULKAN_SUCCESS(res)) {
        WARNING("Failed to allocate descriptor set: %li\n", res);
        status = TEST_VK_DESCRIPTOR_SET_ALLOCATION_ERROR;
        goto cleanup_descriptor_pool;
    }

    input_buffer_info.buffer = input_buffer.buffer;
    input_buffer_info.offset = 0;
    input_buffer_info.range = VK_WHOLE_SIZE;
    output_buffer_info.buffer = output_buffer.buffer;
    output_buffer_info.offset = 0;
    output_buffer_info.range = VK_WHOLE_SIZE;
    descriptor_writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptor_writes[0].dstSet = descriptor_set;
    descriptor_writes[0].dstBinding = 0;
    descriptor_writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptor_writes[0].descriptorCount = 1;
    descriptor_writes[0].pBufferInfo = &input_buffer_info;
    descriptor_writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptor_writes[1].dstSet = descriptor_set;
    descriptor_writes[1].dstBinding = 1;
    descriptor_writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptor_writes[1].descriptorCount = 1;
    descriptor_writes[1].pBufferInfo = &output_buffer_info;
    vkUpdateDescriptorSets(device.device, 2, descriptor_writes, 0, NULL);

    memcpy(shader_words.data(), __comp_spv, __comp_spv_len);
    shader_module_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    shader_module_info.codeSize = __comp_spv_len;
    shader_module_info.pCode = shader_words.data();
    res = vkCreateShaderModule(device.device, &shader_module_info, NULL, &shader_module);
    if (!VULKAN_SUCCESS(res)) {
        WARNING("Failed to create shader module: %li\n", res);
        status = TEST_VK_SHADER_MODULE_CREATION_ERROR;
        goto cleanup_descriptor_pool;
    }

    vkGetPhysicalDeviceProperties(physical_device->physical_device, &properties);
    shared_memory_size = properties.limits.maxComputeSharedMemorySize;
    specialization_entry.constantID = 0;
    specialization_entry.offset = 0;
    specialization_entry.size = sizeof(uint32_t);
    specialization_info.mapEntryCount = 1;
    specialization_info.pMapEntries = &specialization_entry;
    specialization_info.dataSize = sizeof(uint32_t);
    specialization_info.pData = &shared_memory_size;

    pipeline_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipeline_layout_info.setLayoutCount = 1;
    pipeline_layout_info.pSetLayouts = &descriptor_set_layout;
    res = vkCreatePipelineLayout(device.device, &pipeline_layout_info, NULL, &pipeline_layout);
    if (!VULKAN_SUCCESS(res)) {
        WARNING("Failed to create pipeline layout: %li\n", res);
        status = TEST_VK_PIPELINE_LAYOUT_CREATION_ERROR;
        goto cleanup_shader_module;
    }

    pipeline_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipeline_info.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipeline_info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipeline_info.stage.module = shader_module;
    pipeline_info.stage.pName = "main";
    pipeline_info.stage.pSpecializationInfo = &specialization_info;
    pipeline_info.layout = pipeline_layout;
    res = vkCreateComputePipelines(device.device, VK_NULL_HANDLE, 1, &pipeline_info, NULL, &pipeline);
    if (!VULKAN_SUCCESS(res)) {
        WARNING("Failed to create compute pipeline: %li\n", res);
        status = TEST_VK_COMPUTE_PIPELINE_CREATION_ERROR;
        goto cleanup_pipeline_layout;
    }

    command_buffer_alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    command_buffer_alloc_info.commandPool = command_pool;
    command_buffer_alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    command_buffer_alloc_info.commandBufferCount = 1;
    res = vkAllocateCommandBuffers(device.device, &command_buffer_alloc_info, &command_buffer);
    if (!VULKAN_SUCCESS(res)) {
        WARNING("Failed to allocate command buffer: %li\n", res);
        status = TEST_VK_COMMAND_BUFFER_ALLOCATION_ERROR;
        goto cleanup_pipeline;
    }

    INFO("Device Name: %s\n", properties.deviceName);
    INFO("Max Compute Shared Memory: %u bytes\n", shared_memory_size);
    INFO("Timestamp Period: %.6f ns\n", properties.limits.timestampPeriod);
    if (MainGetTestResultFormat() == test_result_csv) {
        LOG_PLAIN("%s,\n", properties.deviceName);
        LOG_PLAIN("WorkGroups,Time_ns\n");
    }

    for (uint32_t workgroups = 1; workgroups <= VULKAN_CUPROBE_WORKGROUP_COUNT_MAX; workgroups++) {
        res = vkResetCommandBuffer(command_buffer, 0);
        if (!VULKAN_SUCCESS(res)) {
            WARNING("Failed to reset command buffer: %li\n", res);
            status = TEST_VK_COMMAND_BUFFER_RESET_ERROR;
            goto cleanup_pipeline;
        }

        VkCommandBufferBeginInfo begin_info = {};
        begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        res = vkBeginCommandBuffer(command_buffer, &begin_info);
        if (!VULKAN_SUCCESS(res)) {
            WARNING("Failed to begin command buffer: %li\n", res);
            status = TEST_VK_COMMAND_BUFFER_BEGIN_ERROR;
            goto cleanup_pipeline;
        }

        vkCmdResetQueryPool(command_buffer, query_pool, 0, 2);
        vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout, 0, 1, &descriptor_set, 0, NULL);
        vkCmdWriteTimestamp(command_buffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, query_pool, 0);
        vkCmdDispatch(command_buffer, workgroups, 1, 1);

        VkMemoryBarrier memory_barrier = {};
        memory_barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        memory_barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        memory_barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
        vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, 1, &memory_barrier, 0, NULL, 0, NULL);
        vkCmdWriteTimestamp(command_buffer, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, query_pool, 1);

        res = vkEndCommandBuffer(command_buffer);
        if (!VULKAN_SUCCESS(res)) {
            WARNING("Failed to end command buffer: %li\n", res);
            status = TEST_VK_COMMAND_BUFFER_END_ERROR;
            goto cleanup_pipeline;
        }

        VkSubmitInfo submit_info = {};
        submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit_info.commandBufferCount = 1;
        submit_info.pCommandBuffers = &command_buffer;
        res = vkQueueSubmit(queue, 1, &submit_info, VK_NULL_HANDLE);
        if (!VULKAN_SUCCESS(res)) {
            WARNING("Failed to submit queue: %li\n", res);
            status = TEST_VK_QUEUE_SUBMIT_ERROR;
            goto cleanup_pipeline;
        }

        res = vkQueueWaitIdle(queue);
        if (!VULKAN_SUCCESS(res)) {
            WARNING("Failed to wait for queue idle: %li\n", res);
            status = TEST_VK_WAIT_FOR_FENCES_ERROR;
            goto cleanup_pipeline;
        }

        uint64_t timestamps[2] = {};
        res = vkGetQueryPoolResults(device.device, query_pool, 0, 2, sizeof(timestamps), timestamps, sizeof(uint64_t), VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
        if (!VULKAN_SUCCESS(res)) {
            WARNING("Failed to get query pool results: %li\n", res);
            status = TEST_UNKNOWN_ERROR;
            goto cleanup_pipeline;
        }

        const double time_ns = (double)(timestamps[1] - timestamps[0]) * properties.limits.timestampPeriod;
        timings.push_back(time_ns);
        if (MainGetTestResultFormat() == test_result_csv) {
            LOG_PLAIN("%u,%.3f\n", workgroups, time_ns);
        } else if (MainGetTestResultFormat() == test_result_raw) {
            LOG_RESULT(workgroups, "%u", "%.3f", workgroups, time_ns);
        } else {
            INFO("WorkGroups %u: %.3f ns\n", workgroups, time_ns);
        }
    }

    if (timings.size() >= 5) {
        double baseline = 0.0;
        for (size_t i = 1; i < 5; i++) {
            baseline += timings[i];
        }
        baseline /= 4.0;

        int detected_cu = -1;
        for (size_t i = 0; i < timings.size(); i++) {
            if (timings[i] > baseline * 1.5) {
                detected_cu = (int)i;
                break;
            }
        }
        if (detected_cu >= 0) {
            INFO("Detected step at workgroup count: %d\n", detected_cu + 1);
            INFO("Estimated compute units: %d\n", detected_cu);
        } else {
            INFO("Could not detect a clear compute-unit step\n");
        }
    }

cleanup_pipeline:
    if (pipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(device.device, pipeline, NULL);
    }
cleanup_pipeline_layout:
    if (pipeline_layout != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(device.device, pipeline_layout, NULL);
    }
cleanup_shader_module:
    if (shader_module != VK_NULL_HANDLE) {
        vkDestroyShaderModule(device.device, shader_module, NULL);
    }
cleanup_descriptor_pool:
    if (descriptor_pool != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(device.device, descriptor_pool, NULL);
    }
cleanup_descriptor_set_layout:
    if (descriptor_set_layout != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(device.device, descriptor_set_layout, NULL);
    }
cleanup_output_buffer:
    _VulkanCUProbeDestroyBuffer(&device, &output_buffer);
cleanup_input_buffer:
    _VulkanCUProbeDestroyBuffer(&device, &input_buffer);
cleanup_command_pool:
    if (command_pool != VK_NULL_HANDLE) {
        vkDestroyCommandPool(device.device, command_pool, NULL);
    }
cleanup_query_pool:
    if (query_pool != VK_NULL_HANDLE) {
        vkDestroyQueryPool(device.device, query_pool, NULL);
    }
cleanup_device:
    VulkanDestroyDevice(&device);
cleanup_queue_properties:
    free(queue_family_properties);
    return status;
}

static test_status _VulkanCUProbeCreateBuffer(vulkan_device *device, VkDeviceSize size, vulkan_cuprobe_buffer *buffer) {
    VkBufferCreateInfo buffer_info = {};
    buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buffer_info.size = size;
    buffer_info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VkResult res = vkCreateBuffer(device->device, &buffer_info, NULL, &buffer->buffer);
    if (!VULKAN_SUCCESS(res)) {
        WARNING("Failed to create buffer: %li\n", res);
        return TEST_VK_BUFFER_CREATION_ERROR;
    }

    VkMemoryRequirements memory_requirements = {};
    vkGetBufferMemoryRequirements(device->device, buffer->buffer, &memory_requirements);

    uint32_t memory_type_index = 0;
    test_status status = _VulkanCUProbeFindMemoryType(device->physical_device, memory_requirements.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, &memory_type_index);
    if (!TEST_SUCCESS(status)) {
        vkDestroyBuffer(device->device, buffer->buffer, NULL);
        buffer->buffer = VK_NULL_HANDLE;
        return status;
    }

    VkMemoryAllocateInfo alloc_info = {};
    alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc_info.allocationSize = memory_requirements.size;
    alloc_info.memoryTypeIndex = memory_type_index;
    res = vkAllocateMemory(device->device, &alloc_info, NULL, &buffer->memory);
    if (!VULKAN_SUCCESS(res)) {
        WARNING("Failed to allocate buffer memory: %li\n", res);
        vkDestroyBuffer(device->device, buffer->buffer, NULL);
        buffer->buffer = VK_NULL_HANDLE;
        return TEST_VK_MEMORY_ALLOCATION_ERROR;
    }

    res = vkBindBufferMemory(device->device, buffer->buffer, buffer->memory, 0);
    if (!VULKAN_SUCCESS(res)) {
        WARNING("Failed to bind buffer memory: %li\n", res);
        vkFreeMemory(device->device, buffer->memory, NULL);
        vkDestroyBuffer(device->device, buffer->buffer, NULL);
        buffer->memory = VK_NULL_HANDLE;
        buffer->buffer = VK_NULL_HANDLE;
        return TEST_VK_BUFFER_BIND_MEMORY_ERROR;
    }

    return TEST_OK;
}

static void _VulkanCUProbeDestroyBuffer(vulkan_device *device, vulkan_cuprobe_buffer *buffer) {
    if (buffer->memory != VK_NULL_HANDLE) {
        vkFreeMemory(device->device, buffer->memory, NULL);
        buffer->memory = VK_NULL_HANDLE;
    }
    if (buffer->buffer != VK_NULL_HANDLE) {
        vkDestroyBuffer(device->device, buffer->buffer, NULL);
        buffer->buffer = VK_NULL_HANDLE;
    }
}

static test_status _VulkanCUProbeFindMemoryType(vulkan_physical_device *physical_device, uint32_t type_filter, VkMemoryPropertyFlags properties, uint32_t *memory_type_index) {
    VkPhysicalDeviceMemoryProperties memory_properties = {};
    vkGetPhysicalDeviceMemoryProperties(physical_device->physical_device, &memory_properties);
    for (uint32_t i = 0; i < memory_properties.memoryTypeCount; i++) {
        if ((type_filter & (1u << i)) != 0 && (memory_properties.memoryTypes[i].propertyFlags & properties) == properties) {
            *memory_type_index = i;
            return TEST_OK;
        }
    }
    return TEST_VK_SUITABLE_ALLOCATION_NOT_FOUND;
}

extern "C" test_status TestsVulkanCUProbeRegister() {
    return VulkanRunnerRegisterTest(&_VulkanCUProbeEntry, NULL, TESTS_VULKAN_CUPROBE_NAME, TESTS_VULKAN_CUPROBE_VERSION, false);
}
