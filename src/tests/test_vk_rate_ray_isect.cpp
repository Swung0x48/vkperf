// Copyright (c) 2026, Swung0x48 <swung0x48@outlook.com>
// SPDX-License-Identifier: MIT

#include "main.h"
#include "logger.h"
#include "helper.h"
#include "vulkan_helper.h"
#include "vulkan_runner.h"
#include "vulkan_shader.h"
#include "tests/test_vk_rate_ray_isect.h"

#include <cstring>

#define VULKAN_RATE_RAY_ISECT_WORKGROUP_SIZE          (256u)
#define VULKAN_RATE_RAY_ISECT_STARTING_WORKGROUPS     (1u)
#define VULKAN_RATE_RAY_ISECT_STARTING_LOOP_COUNT     (16u)
#define VULKAN_RATE_RAY_ISECT_TARGET_TIME_US          (150000ull)
#define VULKAN_RATE_RAY_ISECT_MAX_WORKGROUPS          (4096u)
#define VULKAN_RATE_RAY_ISECT_MAX_LOOP_COUNT          (1u << 22)
#define VULKAN_RATE_RAY_ISECT_GEOMETRY_BOX            (0u)
#define VULKAN_RATE_RAY_ISECT_GEOMETRY_TRIANGLE       (1u)
#define VULKAN_RATE_RAY_ISECT_MODE_ALL_HIT            (0u)
#define VULKAN_RATE_RAY_ISECT_MODE_ALL_MISS           (1u)
#define VULKAN_RATE_RAY_ISECT_MODE_RANDOM             (2u)
#define VULKAN_RATE_RAY_ISECT_OP(geometry, mode)      (((geometry) << 8) | (mode))
#define VULKAN_RATE_RAY_ISECT_OP_BOX                  VULKAN_RATE_RAY_ISECT_OP(VULKAN_RATE_RAY_ISECT_GEOMETRY_BOX, VULKAN_RATE_RAY_ISECT_MODE_ALL_HIT)
#define VULKAN_RATE_RAY_ISECT_OP_TRIANGLE             VULKAN_RATE_RAY_ISECT_OP(VULKAN_RATE_RAY_ISECT_GEOMETRY_TRIANGLE, VULKAN_RATE_RAY_ISECT_MODE_ALL_HIT)
#define VULKAN_RATE_RAY_ISECT_RNG_SEED                (0x9e3779b9u)

typedef struct vulkan_rate_ray_isect_params_t {
    uint32_t loop_count;
    uint32_t ray_mode;
    uint32_t seed;
    uint32_t padding;
} vulkan_rate_ray_isect_params;

typedef struct vulkan_rate_ray_isect_buffer_t {
    VkBuffer buffer;
    VkDeviceMemory memory;
    VkDeviceAddress address;
    VkDeviceSize size;
} vulkan_rate_ray_isect_buffer;

typedef struct vulkan_rate_ray_isect_as_t {
    VkAccelerationStructureKHR handle;
    vulkan_rate_ray_isect_buffer storage;
    VkDeviceAddress address;
} vulkan_rate_ray_isect_as;

typedef struct vulkan_rate_ray_isect_context_t {
    vulkan_device device;
    VkQueue queue;
    uint32_t queue_family_index;
    VkCommandPool command_pool;
    VkCommandBuffer command_buffer;
    PFN_vkGetBufferDeviceAddressKHR vkGetBufferDeviceAddressKHR;
    PFN_vkCreateAccelerationStructureKHR vkCreateAccelerationStructureKHR;
    PFN_vkDestroyAccelerationStructureKHR vkDestroyAccelerationStructureKHR;
    PFN_vkGetAccelerationStructureBuildSizesKHR vkGetAccelerationStructureBuildSizesKHR;
    PFN_vkGetAccelerationStructureDeviceAddressKHR vkGetAccelerationStructureDeviceAddressKHR;
    PFN_vkCmdBuildAccelerationStructuresKHR vkCmdBuildAccelerationStructuresKHR;
} vulkan_rate_ray_isect_context;

static test_status _VulkanRateRayIsectEntry(vulkan_physical_device *physical_device, void *config_data);
static test_status _VulkanRateRayIsectCreateDevice(vulkan_physical_device *physical_device, VkQueueFamilyProperties *queue_family_properties, uint32_t queue_family_count, vulkan_rate_ray_isect_context *context);
static void _VulkanRateRayIsectDestroyDevice(vulkan_rate_ray_isect_context *context);
static test_status _VulkanRateRayIsectCreateBuffer(vulkan_rate_ray_isect_context *context, VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, bool device_address, vulkan_rate_ray_isect_buffer *buffer);
static void _VulkanRateRayIsectDestroyBuffer(vulkan_rate_ray_isect_context *context, vulkan_rate_ray_isect_buffer *buffer);
static test_status _VulkanRateRayIsectFindMemoryType(vulkan_physical_device *physical_device, uint32_t type_filter, VkMemoryPropertyFlags properties, uint32_t *memory_type_index);
static test_status _VulkanRateRayIsectSubmit(vulkan_rate_ray_isect_context *context);
static test_status _VulkanRateRayIsectBuildBLAS(vulkan_rate_ray_isect_context *context, uint32_t operation, vulkan_rate_ray_isect_as *blas, vulkan_rate_ray_isect_buffer *geometry_buffer);
static test_status _VulkanRateRayIsectBuildTLAS(vulkan_rate_ray_isect_context *context, vulkan_rate_ray_isect_as *blas, vulkan_rate_ray_isect_as *tlas, vulkan_rate_ray_isect_buffer *instance_buffer);
static void _VulkanRateRayIsectDestroyAS(vulkan_rate_ray_isect_context *context, vulkan_rate_ray_isect_as *acceleration_structure);
static test_status _VulkanRateRayIsectCreatePipeline(vulkan_rate_ray_isect_context *context, const char *shader_name, VkDescriptorSetLayout *descriptor_set_layout, VkDescriptorPool *descriptor_pool, VkDescriptorSet *descriptor_set, VkPipelineLayout *pipeline_layout, VkPipeline *pipeline, vulkan_shader *shader);
static test_status _VulkanRateRayIsectUpdateDescriptors(vulkan_rate_ray_isect_context *context, VkDescriptorSet descriptor_set, vulkan_rate_ray_isect_as *tlas, vulkan_rate_ray_isect_buffer *uniform_buffer, vulkan_rate_ray_isect_buffer *output_buffer);
static test_status _VulkanRateRayIsectExecute(vulkan_rate_ray_isect_context *context, VkPipelineLayout pipeline_layout, VkPipeline pipeline, VkDescriptorSet descriptor_set, vulkan_rate_ray_isect_buffer *uniform_buffer, uint32_t workgroups, uint32_t loop_count, uint32_t ray_mode, uint64_t *result, uint64_t *time_taken);
static uint32_t _VulkanRateRayIsectGetGeometry(uint32_t operation);
static uint32_t _VulkanRateRayIsectGetMode(uint32_t operation);
static const char *_VulkanRateRayIsectGetShaderName(uint32_t operation);
static const char *_VulkanRateRayIsectGetOperationName(uint32_t operation);
static test_status _VulkanRateRayIsectRegisterSubtest(uint32_t operation, const char *name);

test_status TestsVulkanRateRayIsectRegister() {
    test_status status = _VulkanRateRayIsectRegisterSubtest(VULKAN_RATE_RAY_ISECT_OP_BOX, TESTS_VULKAN_RATE_RAYBOX_ISECT_NAME);
    TEST_RETFAIL(status);
    status = _VulkanRateRayIsectRegisterSubtest(VULKAN_RATE_RAY_ISECT_OP_TRIANGLE, TESTS_VULKAN_RATE_RAYTRI_ISECT_NAME);
    TEST_RETFAIL(status);
    status = _VulkanRateRayIsectRegisterSubtest(VULKAN_RATE_RAY_ISECT_OP(VULKAN_RATE_RAY_ISECT_GEOMETRY_BOX, VULKAN_RATE_RAY_ISECT_MODE_ALL_HIT), TESTS_VULKAN_RATE_RAYBOX_ISECT_ALL_HIT_NAME);
    TEST_RETFAIL(status);
    status = _VulkanRateRayIsectRegisterSubtest(VULKAN_RATE_RAY_ISECT_OP(VULKAN_RATE_RAY_ISECT_GEOMETRY_BOX, VULKAN_RATE_RAY_ISECT_MODE_ALL_MISS), TESTS_VULKAN_RATE_RAYBOX_ISECT_ALL_MISS_NAME);
    TEST_RETFAIL(status);
    status = _VulkanRateRayIsectRegisterSubtest(VULKAN_RATE_RAY_ISECT_OP(VULKAN_RATE_RAY_ISECT_GEOMETRY_BOX, VULKAN_RATE_RAY_ISECT_MODE_RANDOM), TESTS_VULKAN_RATE_RAYBOX_ISECT_RANDOM_NAME);
    TEST_RETFAIL(status);
    status = _VulkanRateRayIsectRegisterSubtest(VULKAN_RATE_RAY_ISECT_OP(VULKAN_RATE_RAY_ISECT_GEOMETRY_TRIANGLE, VULKAN_RATE_RAY_ISECT_MODE_ALL_HIT), TESTS_VULKAN_RATE_RAYTRI_ISECT_ALL_HIT_NAME);
    TEST_RETFAIL(status);
    status = _VulkanRateRayIsectRegisterSubtest(VULKAN_RATE_RAY_ISECT_OP(VULKAN_RATE_RAY_ISECT_GEOMETRY_TRIANGLE, VULKAN_RATE_RAY_ISECT_MODE_ALL_MISS), TESTS_VULKAN_RATE_RAYTRI_ISECT_ALL_MISS_NAME);
    TEST_RETFAIL(status);
    return _VulkanRateRayIsectRegisterSubtest(VULKAN_RATE_RAY_ISECT_OP(VULKAN_RATE_RAY_ISECT_GEOMETRY_TRIANGLE, VULKAN_RATE_RAY_ISECT_MODE_RANDOM), TESTS_VULKAN_RATE_RAYTRI_ISECT_RANDOM_NAME);
}

static test_status _VulkanRateRayIsectRegisterSubtest(uint32_t operation, const char *name) {
    return VulkanRunnerRegisterTest(&_VulkanRateRayIsectEntry, (void *)(uint64_t)operation, name, TESTS_VULKAN_RATE_RAY_ISECT_VERSION, false);
}

static test_status _VulkanRateRayIsectEntry(vulkan_physical_device *physical_device, void *config_data) {
    uint32_t operation = (uint32_t)(uint64_t)config_data;
    uint32_t geometry = _VulkanRateRayIsectGetGeometry(operation);
    uint32_t ray_mode = _VulkanRateRayIsectGetMode(operation);
    const char *shader_name = _VulkanRateRayIsectGetShaderName(operation);
    const char *operation_name = _VulkanRateRayIsectGetOperationName(operation);
    if (shader_name == NULL || operation_name == NULL) {
        return TEST_PROGRAMMING_ERROR;
    }

    test_status status = TEST_OK;
    VkQueueFamilyProperties *queue_family_properties = NULL;
    uint32_t queue_family_count = 0;
    vulkan_rate_ray_isect_context context = {};
    vulkan_rate_ray_isect_as blas = {};
    vulkan_rate_ray_isect_as tlas = {};
    vulkan_rate_ray_isect_buffer geometry_buffer = {};
    vulkan_rate_ray_isect_buffer instance_buffer = {};
    vulkan_rate_ray_isect_buffer uniform_buffer = {};
    vulkan_rate_ray_isect_buffer output_buffer = {};
    VkDescriptorSetLayout descriptor_set_layout = VK_NULL_HANDLE;
    VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
    VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
    VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    vulkan_shader shader = {};
    uint64_t top_result = 0;
    uint32_t top_workgroups = 0;
    uint32_t top_loops = 0;
    uint32_t workgroups = VULKAN_RATE_RAY_ISECT_STARTING_WORKGROUPS;

    INFO("Device Name: %s\n", physical_device->physical_properties.properties.deviceName);

    status = VulkanGetPhysicalQueueFamilyProperties(physical_device, &queue_family_properties, &queue_family_count);
    TEST_RETFAIL(status);
    status = _VulkanRateRayIsectCreateDevice(physical_device, queue_family_properties, queue_family_count, &context);
    if (!TEST_SUCCESS(status)) {
        goto cleanup_queue_properties;
    }

    status = _VulkanRateRayIsectBuildBLAS(&context, geometry, &blas, &geometry_buffer);
    if (!TEST_SUCCESS(status)) {
        goto cleanup_context;
    }
    status = _VulkanRateRayIsectBuildTLAS(&context, &blas, &tlas, &instance_buffer);
    if (!TEST_SUCCESS(status)) {
        goto cleanup_blas;
    }

    status = _VulkanRateRayIsectCreateBuffer(&context, sizeof(vulkan_rate_ray_isect_params), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, false, &uniform_buffer);
    if (!TEST_SUCCESS(status)) {
        goto cleanup_tlas;
    }
    status = _VulkanRateRayIsectCreateBuffer(&context, (VkDeviceSize)VULKAN_RATE_RAY_ISECT_MAX_WORKGROUPS * VULKAN_RATE_RAY_ISECT_WORKGROUP_SIZE * sizeof(uint32_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, false, &output_buffer);
    if (!TEST_SUCCESS(status)) {
        goto cleanup_uniform_buffer;
    }
    status = _VulkanRateRayIsectCreatePipeline(&context, shader_name, &descriptor_set_layout, &descriptor_pool, &descriptor_set, &pipeline_layout, &pipeline, &shader);
    if (!TEST_SUCCESS(status)) {
        goto cleanup_output_buffer;
    }
    status = _VulkanRateRayIsectUpdateDescriptors(&context, descriptor_set, &tlas, &uniform_buffer, &output_buffer);
    if (!TEST_SUCCESS(status)) {
        goto cleanup_pipeline;
    }

    if (MainGetTestResultFormat() == test_result_csv) {
        LOG_PLAIN("%s,\n", physical_device->physical_properties.properties.deviceName);
        LOG_PLAIN("Operation,Ray Intersection Rate (GRay/s),Workgroups,Loops\n");
    }

    while (workgroups <= VULKAN_RATE_RAY_ISECT_MAX_WORKGROUPS) {
        uint32_t loop_count = VULKAN_RATE_RAY_ISECT_STARTING_LOOP_COUNT;
        uint64_t time_taken = 0;
        while (time_taken < VULKAN_RATE_RAY_ISECT_TARGET_TIME_US && loop_count <= VULKAN_RATE_RAY_ISECT_MAX_LOOP_COUNT) {
            uint64_t result = 0;
            status = _VulkanRateRayIsectExecute(&context, pipeline_layout, pipeline, descriptor_set, &uniform_buffer, workgroups, loop_count, ray_mode, &result, &time_taken);
            if (!TEST_SUCCESS(status)) {
                goto cleanup_pipeline;
            }
            if (result > top_result) {
                top_result = result;
                top_workgroups = workgroups;
                top_loops = loop_count;
            }
            if (time_taken < VULKAN_RATE_RAY_ISECT_TARGET_TIME_US) {
                loop_count *= 2;
            }
        }
        if (loop_count == VULKAN_RATE_RAY_ISECT_STARTING_LOOP_COUNT) {
            break;
        }
        workgroups *= 2;
    }

    if (MainGetTestResultFormat() == test_result_csv) {
        LOG_PLAIN("%s,%.3f,%u,%u\n", operation_name, (double)top_result / 1000000000.0, top_workgroups, top_loops);
    } else if (MainGetTestResultFormat() == test_result_raw) {
        LOG_RESULT(0, "%u", "%llu", operation, top_result);
    } else {
        helper_unit_pair unit_conversion;
        HelperConvertUnitsPlain1000(top_result, &unit_conversion);
        INFO("%s rate: %.3f %sRay/s (workgroups: %u, loops: %u)\n", operation_name, unit_conversion.value, unit_conversion.units, top_workgroups, top_loops);
    }

cleanup_pipeline:
    if (pipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(context.device.device, pipeline, NULL);
    }
    if (pipeline_layout != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(context.device.device, pipeline_layout, NULL);
    }
    VulkanShaderCleanUp(&shader);
    if (descriptor_pool != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(context.device.device, descriptor_pool, NULL);
    }
    if (descriptor_set_layout != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(context.device.device, descriptor_set_layout, NULL);
    }
cleanup_output_buffer:
    _VulkanRateRayIsectDestroyBuffer(&context, &output_buffer);
cleanup_uniform_buffer:
    _VulkanRateRayIsectDestroyBuffer(&context, &uniform_buffer);
cleanup_tlas:
    _VulkanRateRayIsectDestroyAS(&context, &tlas);
    _VulkanRateRayIsectDestroyBuffer(&context, &instance_buffer);
cleanup_blas:
    _VulkanRateRayIsectDestroyAS(&context, &blas);
    _VulkanRateRayIsectDestroyBuffer(&context, &geometry_buffer);
cleanup_context:
    _VulkanRateRayIsectDestroyDevice(&context);
cleanup_queue_properties:
    free(queue_family_properties);
    return status;
}

static test_status _VulkanRateRayIsectCreateDevice(vulkan_physical_device *physical_device, VkQueueFamilyProperties *queue_family_properties, uint32_t queue_family_count, vulkan_rate_ray_isect_context *context) {
    test_status status = VulkanSelectQueueFamily(&(context->queue_family_index), queue_family_properties, queue_family_count, VK_QUEUE_COMPUTE_BIT);
    TEST_RETFAIL(status);

    VkExtensionProperties *extensions = NULL;
    uint32_t extension_count = 0;
    status = VulkanGetSupportedExtensions(physical_device, &extensions, &extension_count);
    TEST_RETFAIL(status);

    const char *required_extensions[] = {
        VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
        VK_KHR_RAY_QUERY_EXTENSION_NAME,
        VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME,
        VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME
    };
    for (uint32_t i = 0; i < sizeof(required_extensions) / sizeof(required_extensions[0]); i++) {
        if (!VulkanIsExtensionSupported(required_extensions[i], extensions, extension_count)) {
            WARNING("Required ray tracing extension is not supported: %s\n", required_extensions[i]);
            free(extensions);
            return TEST_VK_FEATURE_UNSUPPORTED;
        }
    }
    free(extensions);

    VkPhysicalDeviceBufferDeviceAddressFeatures buffer_device_address_features = {};
    buffer_device_address_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES;
    VkPhysicalDeviceAccelerationStructureFeaturesKHR acceleration_structure_features = {};
    acceleration_structure_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR;
    acceleration_structure_features.pNext = &buffer_device_address_features;
    VkPhysicalDeviceRayQueryFeaturesKHR ray_query_features = {};
    ray_query_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR;
    ray_query_features.pNext = &acceleration_structure_features;
    VkPhysicalDeviceFeatures2 features = {};
    features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    features.pNext = &ray_query_features;
    vkGetPhysicalDeviceFeatures2(physical_device->physical_device, &features);

    if (!ray_query_features.rayQuery || !acceleration_structure_features.accelerationStructure || !buffer_device_address_features.bufferDeviceAddress) {
        WARNING("Required ray tracing features are not supported\n");
        return TEST_VK_FEATURE_UNSUPPORTED;
    }

    float queue_priority = 1.0f;
    VkDeviceQueueCreateInfo queue_create_info = {};
    queue_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queue_create_info.queueFamilyIndex = context->queue_family_index;
    queue_create_info.queueCount = 1;
    queue_create_info.pQueuePriorities = &queue_priority;

    VkDeviceCreateInfo device_create_info = {};
    device_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    device_create_info.pNext = &ray_query_features;
    device_create_info.queueCreateInfoCount = 1;
    device_create_info.pQueueCreateInfos = &queue_create_info;
    device_create_info.enabledExtensionCount = (uint32_t)(sizeof(required_extensions) / sizeof(required_extensions[0]));
    device_create_info.ppEnabledExtensionNames = required_extensions;

    VkResult res = vkCreateDevice(physical_device->physical_device, &device_create_info, NULL, &(context->device.device));
    if (!VULKAN_SUCCESS(res)) {
        WARNING("Failed to create ray tracing device: %li\n", res);
        return TEST_VK_DEVICE_CREATE_ERROR;
    }

    context->device.physical_device = physical_device;
    context->device.queue_family_count = 1;
    context->device.queue_families = (vulkan_queue_family *)malloc(sizeof(vulkan_queue_family));
    context->device.queues = (VkQueue *)malloc(sizeof(VkQueue));
    if (context->device.queue_families == NULL || context->device.queues == NULL) {
        _VulkanRateRayIsectDestroyDevice(context);
        return TEST_OUT_OF_MEMORY;
    }
    context->device.total_queue_count = 1;
    context->device.queue_families[0].family_index = context->queue_family_index;
    context->device.queue_families[0].queue_offset = 0;
    context->device.queue_families[0].queue_count = 1;
    vkGetDeviceQueue(context->device.device, context->queue_family_index, 0, &(context->queue));
    context->device.queues[0] = context->queue;

    context->vkGetBufferDeviceAddressKHR = (PFN_vkGetBufferDeviceAddressKHR)vkGetDeviceProcAddr(context->device.device, "vkGetBufferDeviceAddressKHR");
    context->vkCreateAccelerationStructureKHR = (PFN_vkCreateAccelerationStructureKHR)vkGetDeviceProcAddr(context->device.device, "vkCreateAccelerationStructureKHR");
    context->vkDestroyAccelerationStructureKHR = (PFN_vkDestroyAccelerationStructureKHR)vkGetDeviceProcAddr(context->device.device, "vkDestroyAccelerationStructureKHR");
    context->vkGetAccelerationStructureBuildSizesKHR = (PFN_vkGetAccelerationStructureBuildSizesKHR)vkGetDeviceProcAddr(context->device.device, "vkGetAccelerationStructureBuildSizesKHR");
    context->vkGetAccelerationStructureDeviceAddressKHR = (PFN_vkGetAccelerationStructureDeviceAddressKHR)vkGetDeviceProcAddr(context->device.device, "vkGetAccelerationStructureDeviceAddressKHR");
    context->vkCmdBuildAccelerationStructuresKHR = (PFN_vkCmdBuildAccelerationStructuresKHR)vkGetDeviceProcAddr(context->device.device, "vkCmdBuildAccelerationStructuresKHR");
    if (context->vkGetBufferDeviceAddressKHR == NULL || context->vkCreateAccelerationStructureKHR == NULL || context->vkDestroyAccelerationStructureKHR == NULL ||
        context->vkGetAccelerationStructureBuildSizesKHR == NULL || context->vkGetAccelerationStructureDeviceAddressKHR == NULL || context->vkCmdBuildAccelerationStructuresKHR == NULL) {
        WARNING("Failed to load ray tracing device functions\n");
        _VulkanRateRayIsectDestroyDevice(context);
        return TEST_VK_FEATURE_UNSUPPORTED;
    }

    VkCommandPoolCreateInfo command_pool_info = {};
    command_pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    command_pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    command_pool_info.queueFamilyIndex = context->queue_family_index;
    res = vkCreateCommandPool(context->device.device, &command_pool_info, NULL, &(context->command_pool));
    if (!VULKAN_SUCCESS(res)) {
        _VulkanRateRayIsectDestroyDevice(context);
        return TEST_VK_COMMAND_POOL_CREATION_ERROR;
    }

    VkCommandBufferAllocateInfo command_buffer_info = {};
    command_buffer_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    command_buffer_info.commandPool = context->command_pool;
    command_buffer_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    command_buffer_info.commandBufferCount = 1;
    res = vkAllocateCommandBuffers(context->device.device, &command_buffer_info, &(context->command_buffer));
    if (!VULKAN_SUCCESS(res)) {
        _VulkanRateRayIsectDestroyDevice(context);
        return TEST_VK_COMMAND_BUFFER_ALLOCATION_ERROR;
    }
    return TEST_OK;
}

static void _VulkanRateRayIsectDestroyDevice(vulkan_rate_ray_isect_context *context) {
    if (context->device.device != VK_NULL_HANDLE) {
        if (context->command_pool != VK_NULL_HANDLE) {
            vkDestroyCommandPool(context->device.device, context->command_pool, NULL);
        }
        vkDestroyDevice(context->device.device, NULL);
    }
    free(context->device.queues);
    free(context->device.queue_families);
    memset(context, 0, sizeof(*context));
}

static test_status _VulkanRateRayIsectCreateBuffer(vulkan_rate_ray_isect_context *context, VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, bool device_address, vulkan_rate_ray_isect_buffer *buffer) {
    memset(buffer, 0, sizeof(*buffer));
    buffer->size = size;
    if (device_address) {
        usage |= VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    }

    VkBufferCreateInfo buffer_info = {};
    buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buffer_info.size = size;
    buffer_info.usage = usage;
    buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VkResult res = vkCreateBuffer(context->device.device, &buffer_info, NULL, &(buffer->buffer));
    if (!VULKAN_SUCCESS(res)) {
        return TEST_VK_BUFFER_CREATION_ERROR;
    }

    VkMemoryRequirements memory_requirements = {};
    VkMemoryAllocateFlagsInfo allocate_flags = {};
    VkMemoryAllocateInfo allocate_info = {};
    vkGetBufferMemoryRequirements(context->device.device, buffer->buffer, &memory_requirements);
    uint32_t memory_type_index = 0;
    test_status status = _VulkanRateRayIsectFindMemoryType(context->device.physical_device, memory_requirements.memoryTypeBits, properties, &memory_type_index);
    if (!TEST_SUCCESS(status)) {
        goto cleanup_buffer;
    }

    allocate_flags.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO;
    allocate_flags.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
    allocate_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocate_info.pNext = device_address ? &allocate_flags : NULL;
    allocate_info.allocationSize = memory_requirements.size;
    allocate_info.memoryTypeIndex = memory_type_index;
    res = vkAllocateMemory(context->device.device, &allocate_info, NULL, &(buffer->memory));
    if (!VULKAN_SUCCESS(res)) {
        status = TEST_VK_MEMORY_ALLOCATION_ERROR;
        goto cleanup_buffer;
    }
    res = vkBindBufferMemory(context->device.device, buffer->buffer, buffer->memory, 0);
    if (!VULKAN_SUCCESS(res)) {
        status = TEST_VK_BUFFER_BIND_MEMORY_ERROR;
        goto cleanup_memory;
    }
    if (device_address) {
        VkBufferDeviceAddressInfo address_info = {};
        address_info.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
        address_info.buffer = buffer->buffer;
        buffer->address = context->vkGetBufferDeviceAddressKHR(context->device.device, &address_info);
        if (buffer->address == 0) {
            status = TEST_UNKNOWN_ERROR;
            goto cleanup_memory;
        }
    }
    return TEST_OK;

cleanup_memory:
    vkFreeMemory(context->device.device, buffer->memory, NULL);
cleanup_buffer:
    vkDestroyBuffer(context->device.device, buffer->buffer, NULL);
    memset(buffer, 0, sizeof(*buffer));
    return status;
}

static void _VulkanRateRayIsectDestroyBuffer(vulkan_rate_ray_isect_context *context, vulkan_rate_ray_isect_buffer *buffer) {
    if (buffer->buffer != VK_NULL_HANDLE) {
        vkDestroyBuffer(context->device.device, buffer->buffer, NULL);
    }
    if (buffer->memory != VK_NULL_HANDLE) {
        vkFreeMemory(context->device.device, buffer->memory, NULL);
    }
    memset(buffer, 0, sizeof(*buffer));
}

static test_status _VulkanRateRayIsectFindMemoryType(vulkan_physical_device *physical_device, uint32_t type_filter, VkMemoryPropertyFlags properties, uint32_t *memory_type_index) {
    VkPhysicalDeviceMemoryProperties *memory_properties = &(physical_device->physical_memory_properties.memoryProperties);
    for (uint32_t i = 0; i < memory_properties->memoryTypeCount; i++) {
        if ((type_filter & (1u << i)) != 0 && (memory_properties->memoryTypes[i].propertyFlags & properties) == properties) {
            *memory_type_index = i;
            return TEST_OK;
        }
    }
    return TEST_VK_SUITABLE_ALLOCATION_NOT_FOUND;
}

static test_status _VulkanRateRayIsectSubmit(vulkan_rate_ray_isect_context *context) {
    VkResult res = vkEndCommandBuffer(context->command_buffer);
    if (!VULKAN_SUCCESS(res)) {
        return TEST_VK_COMMAND_BUFFER_END_ERROR;
    }
    VkSubmitInfo submit_info = {};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &(context->command_buffer);
    res = vkQueueSubmit(context->queue, 1, &submit_info, VK_NULL_HANDLE);
    if (!VULKAN_SUCCESS(res)) {
        return TEST_VK_QUEUE_SUBMIT_ERROR;
    }
    res = vkQueueWaitIdle(context->queue);
    if (!VULKAN_SUCCESS(res)) {
        return TEST_VK_WAIT_FOR_FENCES_ERROR;
    }
    return TEST_OK;
}

static test_status _VulkanRateRayIsectBuildBLAS(vulkan_rate_ray_isect_context *context, uint32_t geometry_type, vulkan_rate_ray_isect_as *blas, vulkan_rate_ray_isect_buffer *geometry_buffer) {
    test_status status = TEST_OK;
    if (geometry_type == VULKAN_RATE_RAY_ISECT_GEOMETRY_BOX) {
        VkAabbPositionsKHR aabb = {-1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f};
        status = _VulkanRateRayIsectCreateBuffer(context, sizeof(aabb), VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, true, geometry_buffer);
        TEST_RETFAIL(status);
        void *mapped = NULL;
        vkMapMemory(context->device.device, geometry_buffer->memory, 0, sizeof(aabb), 0, &mapped);
        memcpy(mapped, &aabb, sizeof(aabb));
        vkUnmapMemory(context->device.device, geometry_buffer->memory);
    } else {
        float vertices[9] = {
            -1.0f, -1.0f, 0.0f,
             1.0f, -1.0f, 0.0f,
             0.0f,  1.0f, 0.0f
        };
        status = _VulkanRateRayIsectCreateBuffer(context, sizeof(vertices), VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, true, geometry_buffer);
        TEST_RETFAIL(status);
        void *mapped = NULL;
        vkMapMemory(context->device.device, geometry_buffer->memory, 0, sizeof(vertices), 0, &mapped);
        memcpy(mapped, vertices, sizeof(vertices));
        vkUnmapMemory(context->device.device, geometry_buffer->memory);
    }

    VkAccelerationStructureGeometryKHR geometry = {};
    geometry.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
    geometry.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
    if (geometry_type == VULKAN_RATE_RAY_ISECT_GEOMETRY_BOX) {
        geometry.geometryType = VK_GEOMETRY_TYPE_AABBS_KHR;
        geometry.geometry.aabbs.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_AABBS_DATA_KHR;
        geometry.geometry.aabbs.data.deviceAddress = geometry_buffer->address;
        geometry.geometry.aabbs.stride = sizeof(VkAabbPositionsKHR);
    } else {
        geometry.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
        geometry.geometry.triangles.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
        geometry.geometry.triangles.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
        geometry.geometry.triangles.vertexData.deviceAddress = geometry_buffer->address;
        geometry.geometry.triangles.vertexStride = sizeof(float) * 3;
        geometry.geometry.triangles.maxVertex = 2;
        geometry.geometry.triangles.indexType = VK_INDEX_TYPE_NONE_KHR;
    }

    VkAccelerationStructureBuildGeometryInfoKHR build_info = {};
    build_info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    build_info.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    build_info.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
    build_info.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    build_info.geometryCount = 1;
    build_info.pGeometries = &geometry;

    uint32_t primitive_count = 1;
    VkAccelerationStructureBuildSizesInfoKHR build_sizes = {};
    build_sizes.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
    context->vkGetAccelerationStructureBuildSizesKHR(context->device.device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &build_info, &primitive_count, &build_sizes);

    status = _VulkanRateRayIsectCreateBuffer(context, build_sizes.accelerationStructureSize, VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, false, &(blas->storage));
    if (!TEST_SUCCESS(status)) {
        return status;
    }

    VkAccelerationStructureCreateInfoKHR as_info = {};
    as_info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
    as_info.buffer = blas->storage.buffer;
    as_info.size = build_sizes.accelerationStructureSize;
    as_info.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    VkResult res = context->vkCreateAccelerationStructureKHR(context->device.device, &as_info, NULL, &(blas->handle));
    if (!VULKAN_SUCCESS(res)) {
        return TEST_UNKNOWN_ERROR;
    }

    vulkan_rate_ray_isect_buffer scratch = {};
    status = _VulkanRateRayIsectCreateBuffer(context, build_sizes.buildScratchSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, true, &scratch);
    if (!TEST_SUCCESS(status)) {
        return status;
    }

    build_info.dstAccelerationStructure = blas->handle;
    build_info.scratchData.deviceAddress = scratch.address;
    VkAccelerationStructureBuildRangeInfoKHR range_info = {};
    range_info.primitiveCount = primitive_count;
    const VkAccelerationStructureBuildRangeInfoKHR *range_infos[] = {&range_info};

    vkResetCommandBuffer(context->command_buffer, 0);
    VkCommandBufferBeginInfo begin_info = {};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    res = vkBeginCommandBuffer(context->command_buffer, &begin_info);
    if (!VULKAN_SUCCESS(res)) {
        _VulkanRateRayIsectDestroyBuffer(context, &scratch);
        return TEST_VK_COMMAND_BUFFER_BEGIN_ERROR;
    }
    context->vkCmdBuildAccelerationStructuresKHR(context->command_buffer, 1, &build_info, range_infos);
    VkMemoryBarrier barrier = {};
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
    barrier.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
    vkCmdPipelineBarrier(context->command_buffer, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR, 0, 1, &barrier, 0, NULL, 0, NULL);
    status = _VulkanRateRayIsectSubmit(context);
    _VulkanRateRayIsectDestroyBuffer(context, &scratch);
    if (!TEST_SUCCESS(status)) {
        return status;
    }

    VkAccelerationStructureDeviceAddressInfoKHR address_info = {};
    address_info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;
    address_info.accelerationStructure = blas->handle;
    blas->address = context->vkGetAccelerationStructureDeviceAddressKHR(context->device.device, &address_info);
    return blas->address == 0 ? TEST_UNKNOWN_ERROR : TEST_OK;
}

static test_status _VulkanRateRayIsectBuildTLAS(vulkan_rate_ray_isect_context *context, vulkan_rate_ray_isect_as *blas, vulkan_rate_ray_isect_as *tlas, vulkan_rate_ray_isect_buffer *instance_buffer) {
    VkTransformMatrixKHR transform = {{{1.0f, 0.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 1.0f, 0.0f}}};
    VkAccelerationStructureInstanceKHR instance = {};
    instance.transform = transform;
    instance.instanceCustomIndex = 0;
    instance.mask = 0xff;
    instance.instanceShaderBindingTableRecordOffset = 0;
    instance.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
    instance.accelerationStructureReference = blas->address;

    test_status status = _VulkanRateRayIsectCreateBuffer(context, sizeof(instance), VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, true, instance_buffer);
    TEST_RETFAIL(status);
    void *mapped = NULL;
    vkMapMemory(context->device.device, instance_buffer->memory, 0, sizeof(instance), 0, &mapped);
    memcpy(mapped, &instance, sizeof(instance));
    vkUnmapMemory(context->device.device, instance_buffer->memory);

    VkAccelerationStructureGeometryKHR geometry = {};
    geometry.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
    geometry.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
    geometry.geometry.instances.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
    geometry.geometry.instances.arrayOfPointers = VK_FALSE;
    geometry.geometry.instances.data.deviceAddress = instance_buffer->address;

    VkAccelerationStructureBuildGeometryInfoKHR build_info = {};
    build_info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    build_info.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
    build_info.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
    build_info.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    build_info.geometryCount = 1;
    build_info.pGeometries = &geometry;

    uint32_t primitive_count = 1;
    VkAccelerationStructureBuildSizesInfoKHR build_sizes = {};
    build_sizes.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
    context->vkGetAccelerationStructureBuildSizesKHR(context->device.device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &build_info, &primitive_count, &build_sizes);

    status = _VulkanRateRayIsectCreateBuffer(context, build_sizes.accelerationStructureSize, VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, false, &(tlas->storage));
    TEST_RETFAIL(status);
    VkAccelerationStructureCreateInfoKHR as_info = {};
    as_info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
    as_info.buffer = tlas->storage.buffer;
    as_info.size = build_sizes.accelerationStructureSize;
    as_info.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
    VkResult res = context->vkCreateAccelerationStructureKHR(context->device.device, &as_info, NULL, &(tlas->handle));
    if (!VULKAN_SUCCESS(res)) {
        return TEST_UNKNOWN_ERROR;
    }

    vulkan_rate_ray_isect_buffer scratch = {};
    status = _VulkanRateRayIsectCreateBuffer(context, build_sizes.buildScratchSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, true, &scratch);
    TEST_RETFAIL(status);
    build_info.dstAccelerationStructure = tlas->handle;
    build_info.scratchData.deviceAddress = scratch.address;
    VkAccelerationStructureBuildRangeInfoKHR range_info = {};
    range_info.primitiveCount = primitive_count;
    const VkAccelerationStructureBuildRangeInfoKHR *range_infos[] = {&range_info};

    vkResetCommandBuffer(context->command_buffer, 0);
    VkCommandBufferBeginInfo begin_info = {};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    res = vkBeginCommandBuffer(context->command_buffer, &begin_info);
    if (!VULKAN_SUCCESS(res)) {
        _VulkanRateRayIsectDestroyBuffer(context, &scratch);
        return TEST_VK_COMMAND_BUFFER_BEGIN_ERROR;
    }
    context->vkCmdBuildAccelerationStructuresKHR(context->command_buffer, 1, &build_info, range_infos);
    VkMemoryBarrier barrier = {};
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
    barrier.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
    vkCmdPipelineBarrier(context->command_buffer, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, NULL, 0, NULL);
    status = _VulkanRateRayIsectSubmit(context);
    _VulkanRateRayIsectDestroyBuffer(context, &scratch);
    if (!TEST_SUCCESS(status)) {
        return status;
    }

    VkAccelerationStructureDeviceAddressInfoKHR address_info = {};
    address_info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;
    address_info.accelerationStructure = tlas->handle;
    tlas->address = context->vkGetAccelerationStructureDeviceAddressKHR(context->device.device, &address_info);
    return tlas->address == 0 ? TEST_UNKNOWN_ERROR : TEST_OK;
}

static void _VulkanRateRayIsectDestroyAS(vulkan_rate_ray_isect_context *context, vulkan_rate_ray_isect_as *acceleration_structure) {
    if (acceleration_structure->handle != VK_NULL_HANDLE) {
        context->vkDestroyAccelerationStructureKHR(context->device.device, acceleration_structure->handle, NULL);
    }
    _VulkanRateRayIsectDestroyBuffer(context, &(acceleration_structure->storage));
    memset(acceleration_structure, 0, sizeof(*acceleration_structure));
}

static test_status _VulkanRateRayIsectCreatePipeline(vulkan_rate_ray_isect_context *context, const char *shader_name, VkDescriptorSetLayout *descriptor_set_layout, VkDescriptorPool *descriptor_pool, VkDescriptorSet *descriptor_set, VkPipelineLayout *pipeline_layout, VkPipeline *pipeline, vulkan_shader *shader) {
    test_status status = VulkanShaderInitializeFromFile(&(context->device), shader_name, VK_SHADER_STAGE_COMPUTE_BIT, shader);
    TEST_RETFAIL(status);

    VkDescriptorSetLayoutBinding bindings[3] = {};
    bindings[0].binding = 0;
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    bindings[1].binding = 1;
    bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    bindings[1].descriptorCount = 1;
    bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    bindings[2].binding = 2;
    bindings[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[2].descriptorCount = 1;
    bindings[2].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutCreateInfo layout_info = {};
    layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layout_info.bindingCount = 3;
    layout_info.pBindings = bindings;
    VkResult res = vkCreateDescriptorSetLayout(context->device.device, &layout_info, NULL, descriptor_set_layout);
    if (!VULKAN_SUCCESS(res)) {
        return TEST_VK_DESCRIPTOR_SET_LAYOUT_CREATION_ERROR;
    }

    VkDescriptorPoolSize pool_sizes[3] = {};
    pool_sizes[0].type = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
    pool_sizes[0].descriptorCount = 1;
    pool_sizes[1].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    pool_sizes[1].descriptorCount = 1;
    pool_sizes[2].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    pool_sizes[2].descriptorCount = 1;
    VkDescriptorPoolCreateInfo pool_info = {};
    pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool_info.poolSizeCount = 3;
    pool_info.pPoolSizes = pool_sizes;
    pool_info.maxSets = 1;
    res = vkCreateDescriptorPool(context->device.device, &pool_info, NULL, descriptor_pool);
    if (!VULKAN_SUCCESS(res)) {
        return TEST_VK_DESCRIPTOR_POOL_CREATION_ERROR;
    }

    VkDescriptorSetAllocateInfo set_info = {};
    set_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    set_info.descriptorPool = *descriptor_pool;
    set_info.descriptorSetCount = 1;
    set_info.pSetLayouts = descriptor_set_layout;
    res = vkAllocateDescriptorSets(context->device.device, &set_info, descriptor_set);
    if (!VULKAN_SUCCESS(res)) {
        return TEST_VK_DESCRIPTOR_SET_ALLOCATION_ERROR;
    }

    VkPipelineLayoutCreateInfo pipeline_layout_info = {};
    pipeline_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipeline_layout_info.setLayoutCount = 1;
    pipeline_layout_info.pSetLayouts = descriptor_set_layout;
    res = vkCreatePipelineLayout(context->device.device, &pipeline_layout_info, NULL, pipeline_layout);
    if (!VULKAN_SUCCESS(res)) {
        return TEST_VK_PIPELINE_LAYOUT_CREATION_ERROR;
    }

    VkComputePipelineCreateInfo pipeline_info = {};
    pipeline_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipeline_info.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipeline_info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipeline_info.stage.module = shader->shader_module;
    pipeline_info.stage.pName = "main";
    pipeline_info.layout = *pipeline_layout;
    res = vkCreateComputePipelines(context->device.device, VK_NULL_HANDLE, 1, &pipeline_info, NULL, pipeline);
    if (!VULKAN_SUCCESS(res)) {
        WARNING("Failed to create ray query compute pipeline: %li\n", res);
        return TEST_VK_COMPUTE_PIPELINE_CREATION_ERROR;
    }
    return TEST_OK;
}

static test_status _VulkanRateRayIsectUpdateDescriptors(vulkan_rate_ray_isect_context *context, VkDescriptorSet descriptor_set, vulkan_rate_ray_isect_as *tlas, vulkan_rate_ray_isect_buffer *uniform_buffer, vulkan_rate_ray_isect_buffer *output_buffer) {
    VkWriteDescriptorSetAccelerationStructureKHR acceleration_structure_info = {};
    acceleration_structure_info.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR;
    acceleration_structure_info.accelerationStructureCount = 1;
    acceleration_structure_info.pAccelerationStructures = &(tlas->handle);
    VkDescriptorBufferInfo uniform_info = {};
    uniform_info.buffer = uniform_buffer->buffer;
    uniform_info.offset = 0;
    uniform_info.range = sizeof(vulkan_rate_ray_isect_params);
    VkDescriptorBufferInfo output_info = {};
    output_info.buffer = output_buffer->buffer;
    output_info.offset = 0;
    output_info.range = output_buffer->size;

    VkWriteDescriptorSet writes[3] = {};
    writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[0].pNext = &acceleration_structure_info;
    writes[0].dstSet = descriptor_set;
    writes[0].dstBinding = 0;
    writes[0].descriptorCount = 1;
    writes[0].descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
    writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[1].dstSet = descriptor_set;
    writes[1].dstBinding = 1;
    writes[1].descriptorCount = 1;
    writes[1].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    writes[1].pBufferInfo = &uniform_info;
    writes[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[2].dstSet = descriptor_set;
    writes[2].dstBinding = 2;
    writes[2].descriptorCount = 1;
    writes[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[2].pBufferInfo = &output_info;
    vkUpdateDescriptorSets(context->device.device, 3, writes, 0, NULL);
    return TEST_OK;
}

static test_status _VulkanRateRayIsectExecute(vulkan_rate_ray_isect_context *context, VkPipelineLayout pipeline_layout, VkPipeline pipeline, VkDescriptorSet descriptor_set, vulkan_rate_ray_isect_buffer *uniform_buffer, uint32_t workgroups, uint32_t loop_count, uint32_t ray_mode, uint64_t *result, uint64_t *time_taken) {
    void *mapped = NULL;
    VkResult res = vkMapMemory(context->device.device, uniform_buffer->memory, 0, sizeof(vulkan_rate_ray_isect_params), 0, &mapped);
    if (!VULKAN_SUCCESS(res)) {
        return TEST_VK_MEMORY_MAPPING_ERROR;
    }
    vulkan_rate_ray_isect_params params = {};
    params.loop_count = loop_count;
    params.ray_mode = ray_mode;
    params.seed = VULKAN_RATE_RAY_ISECT_RNG_SEED;
    memcpy(mapped, &params, sizeof(params));
    vkUnmapMemory(context->device.device, uniform_buffer->memory);

    uint32_t groups_x = 0;
    uint32_t groups_y = 0;
    uint32_t groups_z = 0;
    test_status status = VulkanCalculateWorkgroupDispatch(&(context->device), workgroups, &groups_x, &groups_y, &groups_z);
    TEST_RETFAIL(status);

    vkResetCommandBuffer(context->command_buffer, 0);
    VkCommandBufferBeginInfo begin_info = {};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    res = vkBeginCommandBuffer(context->command_buffer, &begin_info);
    if (!VULKAN_SUCCESS(res)) {
        return TEST_VK_COMMAND_BUFFER_BEGIN_ERROR;
    }
    vkCmdBindPipeline(context->command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdBindDescriptorSets(context->command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout, 0, 1, &descriptor_set, 0, NULL);
    vkCmdDispatch(context->command_buffer, groups_x, groups_y, groups_z);
    HelperResetTimestamp();
    status = _VulkanRateRayIsectSubmit(context);
    TEST_RETFAIL(status);
    uint64_t time = HelperMarkTimestamp();
    if (time_taken != NULL) {
        *time_taken = time;
    }
    if (time == 0) {
        if (result != NULL) {
            *result = 0;
        }
    } else {
        uint64_t total_intersections = (uint64_t)groups_x * groups_y * groups_z * VULKAN_RATE_RAY_ISECT_WORKGROUP_SIZE * loop_count;
        uint64_t throughput_per_second = (total_intersections * 1000000ull) / time;
        helper_unit_pair unit_conversion;
        HelperConvertUnitsPlain1000(throughput_per_second, &unit_conversion);
        INFO("Loop count %u workgroup count %u took %.3fms (rate: %.3f %sRay/s)\n", loop_count, workgroups, time / 1000.0f, unit_conversion.value, unit_conversion.units);
        if (result != NULL) {
            *result = throughput_per_second;
        }
    }
    return TEST_OK;
}

static uint32_t _VulkanRateRayIsectGetGeometry(uint32_t operation) {
    return operation >> 8;
}

static uint32_t _VulkanRateRayIsectGetMode(uint32_t operation) {
    return operation & 0xffu;
}

static const char *_VulkanRateRayIsectGetShaderName(uint32_t operation) {
    switch (_VulkanRateRayIsectGetGeometry(operation)) {
    case VULKAN_RATE_RAY_ISECT_GEOMETRY_BOX:
        return "vulkan_rate_raybox_isect.spv";
    case VULKAN_RATE_RAY_ISECT_GEOMETRY_TRIANGLE:
        return "vulkan_rate_raytri_isect.spv";
    default:
        return NULL;
    }
}

static const char *_VulkanRateRayIsectGetOperationName(uint32_t operation) {
    switch (_VulkanRateRayIsectGetGeometry(operation)) {
    case VULKAN_RATE_RAY_ISECT_GEOMETRY_BOX:
        switch (_VulkanRateRayIsectGetMode(operation)) {
        case VULKAN_RATE_RAY_ISECT_MODE_ALL_HIT:
            return "rayBoxAllHit";
        case VULKAN_RATE_RAY_ISECT_MODE_ALL_MISS:
            return "rayBoxAllMiss";
        case VULKAN_RATE_RAY_ISECT_MODE_RANDOM:
            return "rayBoxRandom";
        default:
            return NULL;
        }
    case VULKAN_RATE_RAY_ISECT_GEOMETRY_TRIANGLE:
        switch (_VulkanRateRayIsectGetMode(operation)) {
        case VULKAN_RATE_RAY_ISECT_MODE_ALL_HIT:
            return "rayTriangleAllHit";
        case VULKAN_RATE_RAY_ISECT_MODE_ALL_MISS:
            return "rayTriangleAllMiss";
        case VULKAN_RATE_RAY_ISECT_MODE_RANDOM:
            return "rayTriangleRandom";
        default:
            return NULL;
        }
    default:
        return NULL;
    }
}
