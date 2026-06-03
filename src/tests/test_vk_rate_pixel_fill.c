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
#include "tests/test_vk_rate_pixel_fill.h"

#define VULKAN_RATE_PIXEL_FILL_TARGET_TIME_US       (150000)
#define VULKAN_RATE_PIXEL_FILL_WARMUP_RUNS          (3)
#define VULKAN_RATE_PIXEL_FILL_MAX_TARGET_SIDE      (8192)
#define VULKAN_RATE_PIXEL_FILL_MAX_INSTANCES        (65536)
#define VULKAN_RATE_PIXEL_FILL_FORMAT               VK_FORMAT_R8G8B8A8_UNORM

typedef struct vulkan_rate_pixel_fill_pipeline_t {
    vulkan_shader vertex_shader;
    vulkan_shader fragment_shader;
    VkRenderPass render_pass;
    VkPipelineLayout pipeline_layout;
    VkPipeline pipeline;
} vulkan_rate_pixel_fill_pipeline;

typedef struct vulkan_rate_pixel_fill_target_t {
    vulkan_device *device;
    VkImage image;
    VkDeviceMemory memory;
    VkImageView image_view;
    VkFramebuffer framebuffer;
    uint32_t side;
} vulkan_rate_pixel_fill_target;

static test_status _VulkanRatePixelFillEntry(vulkan_physical_device *device, void *config_data);
static test_status _VulkanRatePixelFillCreateRenderPass(vulkan_device *device, VkRenderPass *render_pass);
static test_status _VulkanRatePixelFillCreatePipeline(vulkan_device *device, vulkan_rate_pixel_fill_pipeline *pipeline);
static test_status _VulkanRatePixelFillCleanPipeline(vulkan_device *device, vulkan_rate_pixel_fill_pipeline *pipeline);
static test_status _VulkanRatePixelFillCreateTarget(vulkan_device *device, VkRenderPass render_pass, uint32_t side, vulkan_rate_pixel_fill_target *target);
static test_status _VulkanRatePixelFillCleanTarget(vulkan_rate_pixel_fill_target *target);
static test_status _VulkanRatePixelFillTransitionTarget(vulkan_rate_pixel_fill_target *target, vulkan_command_sequence *command_sequence);
static test_status _VulkanRatePixelFillRunSide(vulkan_device *device, vulkan_rate_pixel_fill_pipeline *pipeline, vulkan_command_sequence *command_sequence, uint32_t side, uint64_t *top_result, uint64_t *top_instances);
static test_status _VulkanRatePixelFillExecute(uint32_t instance_count, uint32_t side, uint64_t *result, uint64_t *time_taken, vulkan_rate_pixel_fill_pipeline *pipeline, vulkan_rate_pixel_fill_target *target, vulkan_command_sequence *command_sequence);
static test_status _VulkanRatePixelFillFindMemoryType(vulkan_physical_device *physical_device, uint32_t type_filter, VkMemoryPropertyFlags properties, uint32_t *memory_type_index);

test_status TestsVulkanRatePixelFillRegister() {
    return VulkanRunnerRegisterTest(&_VulkanRatePixelFillEntry, NULL, TESTS_VULKAN_RATE_PIXEL_FILL_NAME, TESTS_VULKAN_RATE_PIXEL_FILL_VERSION, false);
}

static test_status _VulkanRatePixelFillEntry(vulkan_physical_device *physical_device, void *config_data) {
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
    status = VulkanCreateDevice(physical_device, queue_family_properties, queue_family_count, VK_QUEUE_GRAPHICS_BIT, &device, NULL);
    if (!TEST_SUCCESS(status)) {
        goto cleanup_queue_properties;
    }

    vulkan_rate_pixel_fill_pipeline pipeline;
    memset(&pipeline, 0, sizeof(pipeline));
    status = _VulkanRatePixelFillCreatePipeline(&device, &pipeline);
    if (!TEST_SUCCESS(status)) {
        goto cleanup_device;
    }

    vulkan_command_buffer command_buffer;
    status = VulkanCommandBufferInitialize(&device, 1, &command_buffer);
    if (!TEST_SUCCESS(status)) {
        goto cleanup_pipeline;
    }

    vulkan_command_sequence command_sequence;
    status = VulkanCommandSequenceInitialize(&command_buffer, VULKAN_COMMAND_SEQUENCE_RESET_ON_COMPLETION, &command_sequence);
    if (!TEST_SUCCESS(status)) {
        goto cleanup_command_buffer;
    }

    uint32_t maximum_target_side = min(device.physical_device->physical_properties.properties.limits.maxImageDimension2D, VULKAN_RATE_PIXEL_FILL_MAX_TARGET_SIDE);
    INFO("Maximum render target side tested: %lu\n", maximum_target_side);
    INFO("Final results for %s:\n", physical_device->physical_properties.properties.deviceName);
    if (MainGetTestResultFormat() == test_result_csv) {
        LOG_PLAIN("%s,\n", physical_device->physical_properties.properties.deviceName);
        LOG_PLAIN("Operation,Render Target Edge,Pixels,Pixel Fill Rate (GPixel/s),Instances\n");
    }

    uint32_t result_index = 0;
    for (uint32_t side = 1; side <= maximum_target_side; side <<= 1) {
        uint64_t top_result = 0;
        uint64_t top_instances = 0;

        INFO("Testing pixel fill render target side %lu...\n", side);
        status = _VulkanRatePixelFillRunSide(&device, &pipeline, &command_sequence, side, &top_result, &top_instances);
        if (!TEST_SUCCESS(status)) {
            goto cleanup_command_buffer;
        }

        uint64_t pixels = (uint64_t)side * (uint64_t)side * top_instances;
        if (MainGetTestResultFormat() == test_result_csv) {
            LOG_PLAIN("fill,%lu,%llu,%.3f,%llu\n", side, pixels, (double)top_result / 1000000000.0, top_instances);
        } else if (MainGetTestResultFormat() == test_result_raw) {
            LOG_RESULT(result_index, "%lu", "%llu", side, top_result);
        } else {
            helper_unit_pair pixel_conversion;
            HelperConvertUnitsPlain1000(top_result, &pixel_conversion);
            INFO("Pixel fill for %lux%lu: %.3f %sPixel/s (instances: %llu)\n", side, side, pixel_conversion.value, pixel_conversion.units, top_instances);
        }
        result_index++;
    }

cleanup_command_buffer:
    VulkanCommandBufferCleanUp(&command_buffer);
cleanup_pipeline:
    _VulkanRatePixelFillCleanPipeline(&device, &pipeline);
cleanup_device:
    VulkanDestroyDevice(&device);
cleanup_queue_properties:
    free(queue_family_properties);
error:
    return status;
}

static test_status _VulkanRatePixelFillCreateRenderPass(vulkan_device *device, VkRenderPass *render_pass) {
    VkAttachmentDescription color_attachment = {0};
    color_attachment.format = VULKAN_RATE_PIXEL_FILL_FORMAT;
    color_attachment.samples = VK_SAMPLE_COUNT_1_BIT;
    color_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    color_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    color_attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    color_attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    color_attachment.initialLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    color_attachment.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkAttachmentReference color_attachment_reference = {0};
    color_attachment_reference.attachment = 0;
    color_attachment_reference.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkSubpassDescription subpass = {0};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &color_attachment_reference;

    VkSubpassDependency subpass_dependency = {0};
    subpass_dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
    subpass_dependency.dstSubpass = 0;
    subpass_dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    subpass_dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    subpass_dependency.srcAccessMask = 0;
    subpass_dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

    VkRenderPassCreateInfo render_pass_create_info = {0};
    render_pass_create_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    render_pass_create_info.attachmentCount = 1;
    render_pass_create_info.pAttachments = &color_attachment;
    render_pass_create_info.subpassCount = 1;
    render_pass_create_info.pSubpasses = &subpass;
    render_pass_create_info.dependencyCount = 1;
    render_pass_create_info.pDependencies = &subpass_dependency;

    VkResult res = vkCreateRenderPass(device->device, &render_pass_create_info, NULL, render_pass);
    VULKAN_RETFAIL(res, TEST_UNKNOWN_ERROR);
    return TEST_OK;
}

static test_status _VulkanRatePixelFillCreatePipeline(vulkan_device *device, vulkan_rate_pixel_fill_pipeline *pipeline) {
    test_status status = _VulkanRatePixelFillCreateRenderPass(device, &(pipeline->render_pass));
    TEST_RETFAIL(status);

    status = VulkanShaderInitializeFromFile(device, "vulkan_rate_pixel_fill_vertex.spv", VK_SHADER_STAGE_VERTEX_BIT, &(pipeline->vertex_shader));
    if (!TEST_SUCCESS(status)) {
        goto cleanup_render_pass;
    }
    status = VulkanShaderInitializeFromFile(device, "vulkan_rate_pixel_fill_fragment.spv", VK_SHADER_STAGE_FRAGMENT_BIT, &(pipeline->fragment_shader));
    if (!TEST_SUCCESS(status)) {
        goto cleanup_vertex_shader;
    }

    VkPipelineLayoutCreateInfo pipeline_layout_create_info = {0};
    pipeline_layout_create_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;

    VkResult res = vkCreatePipelineLayout(device->device, &pipeline_layout_create_info, NULL, &(pipeline->pipeline_layout));
    if (!VULKAN_SUCCESS(res)) {
        status = TEST_VK_PIPELINE_LAYOUT_CREATION_ERROR;
        goto cleanup_fragment_shader;
    }

    VkPipelineShaderStageCreateInfo shader_stages[2] = {0};
    shader_stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    shader_stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    shader_stages[0].module = pipeline->vertex_shader.shader_module;
    shader_stages[0].pName = "main";
    shader_stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    shader_stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    shader_stages[1].module = pipeline->fragment_shader.shader_module;
    shader_stages[1].pName = "main";

    VkPipelineVertexInputStateCreateInfo vertex_input_state = {0};
    vertex_input_state.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

    VkPipelineInputAssemblyStateCreateInfo input_assembly_state = {0};
    input_assembly_state.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    input_assembly_state.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkPipelineViewportStateCreateInfo viewport_state = {0};
    viewport_state.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewport_state.viewportCount = 1;
    viewport_state.scissorCount = 1;

    VkPipelineRasterizationStateCreateInfo rasterization_state = {0};
    rasterization_state.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterization_state.polygonMode = VK_POLYGON_MODE_FILL;
    rasterization_state.cullMode = VK_CULL_MODE_NONE;
    rasterization_state.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rasterization_state.lineWidth = 1.0f;

    VkPipelineMultisampleStateCreateInfo multisample_state = {0};
    multisample_state.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisample_state.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineColorBlendAttachmentState color_blend_attachment = {0};
    color_blend_attachment.blendEnable = VK_FALSE;
    color_blend_attachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

    VkPipelineColorBlendStateCreateInfo color_blend_state = {0};
    color_blend_state.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    color_blend_state.attachmentCount = 1;
    color_blend_state.pAttachments = &color_blend_attachment;

    VkDynamicState dynamic_states[2] = {
        VK_DYNAMIC_STATE_VIEWPORT,
        VK_DYNAMIC_STATE_SCISSOR
    };
    VkPipelineDynamicStateCreateInfo dynamic_state = {0};
    dynamic_state.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamic_state.dynamicStateCount = 2;
    dynamic_state.pDynamicStates = dynamic_states;

    VkGraphicsPipelineCreateInfo pipeline_create_info = {0};
    pipeline_create_info.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipeline_create_info.stageCount = 2;
    pipeline_create_info.pStages = shader_stages;
    pipeline_create_info.pVertexInputState = &vertex_input_state;
    pipeline_create_info.pInputAssemblyState = &input_assembly_state;
    pipeline_create_info.pViewportState = &viewport_state;
    pipeline_create_info.pRasterizationState = &rasterization_state;
    pipeline_create_info.pMultisampleState = &multisample_state;
    pipeline_create_info.pColorBlendState = &color_blend_state;
    pipeline_create_info.pDynamicState = &dynamic_state;
    pipeline_create_info.layout = pipeline->pipeline_layout;
    pipeline_create_info.renderPass = pipeline->render_pass;
    pipeline_create_info.subpass = 0;

    res = vkCreateGraphicsPipelines(device->device, VK_NULL_HANDLE, 1, &pipeline_create_info, NULL, &(pipeline->pipeline));
    if (!VULKAN_SUCCESS(res)) {
        WARNING("Encountered a Vulkan error: %li\n", res);
        status = TEST_UNKNOWN_ERROR;
        goto cleanup_pipeline_layout;
    }
    return TEST_OK;

cleanup_pipeline_layout:
    vkDestroyPipelineLayout(device->device, pipeline->pipeline_layout, NULL);
    pipeline->pipeline_layout = VK_NULL_HANDLE;
cleanup_fragment_shader:
    VulkanShaderCleanUp(&(pipeline->fragment_shader));
cleanup_vertex_shader:
    VulkanShaderCleanUp(&(pipeline->vertex_shader));
cleanup_render_pass:
    vkDestroyRenderPass(device->device, pipeline->render_pass, NULL);
    pipeline->render_pass = VK_NULL_HANDLE;
    return status;
}

static test_status _VulkanRatePixelFillCleanPipeline(vulkan_device *device, vulkan_rate_pixel_fill_pipeline *pipeline) {
    if (pipeline->pipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(device->device, pipeline->pipeline, NULL);
    }
    if (pipeline->pipeline_layout != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(device->device, pipeline->pipeline_layout, NULL);
    }
    VulkanShaderCleanUp(&(pipeline->fragment_shader));
    VulkanShaderCleanUp(&(pipeline->vertex_shader));
    if (pipeline->render_pass != VK_NULL_HANDLE) {
        vkDestroyRenderPass(device->device, pipeline->render_pass, NULL);
    }
    return TEST_OK;
}

static test_status _VulkanRatePixelFillCreateTarget(vulkan_device *device, VkRenderPass render_pass, uint32_t side, vulkan_rate_pixel_fill_target *target) {
    memset(target, 0, sizeof(*target));
    target->device = device;
    target->side = side;

    VkImageCreateInfo image_create_info = {0};
    image_create_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    image_create_info.imageType = VK_IMAGE_TYPE_2D;
    image_create_info.format = VULKAN_RATE_PIXEL_FILL_FORMAT;
    image_create_info.extent.width = side;
    image_create_info.extent.height = side;
    image_create_info.extent.depth = 1;
    image_create_info.mipLevels = 1;
    image_create_info.arrayLayers = 1;
    image_create_info.samples = VK_SAMPLE_COUNT_1_BIT;
    image_create_info.tiling = VK_IMAGE_TILING_OPTIMAL;
    image_create_info.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    image_create_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    image_create_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VkResult res = vkCreateImage(device->device, &image_create_info, NULL, &(target->image));
    VULKAN_RETFAIL(res, TEST_VK_IMAGE_CREATION_ERROR);

    VkMemoryRequirements memory_requirements;
    vkGetImageMemoryRequirements(device->device, target->image, &memory_requirements);

    uint32_t memory_type_index = 0;
    test_status status = _VulkanRatePixelFillFindMemoryType(device->physical_device, memory_requirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &memory_type_index);
    if (!TEST_SUCCESS(status)) {
        goto cleanup_image;
    }

    VkMemoryAllocateInfo memory_allocate_info = {0};
    memory_allocate_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    memory_allocate_info.allocationSize = memory_requirements.size;
    memory_allocate_info.memoryTypeIndex = memory_type_index;

    res = vkAllocateMemory(device->device, &memory_allocate_info, NULL, &(target->memory));
    if (!VULKAN_SUCCESS(res)) {
        status = TEST_VK_MEMORY_ALLOCATION_ERROR;
        goto cleanup_image;
    }

    res = vkBindImageMemory(device->device, target->image, target->memory, 0);
    if (!VULKAN_SUCCESS(res)) {
        status = TEST_VK_IMAGE_BIND_MEMORY_ERROR;
        goto cleanup_image_with_memory;
    }

    VkImageViewCreateInfo image_view_create_info = {0};
    image_view_create_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    image_view_create_info.image = target->image;
    image_view_create_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
    image_view_create_info.format = VULKAN_RATE_PIXEL_FILL_FORMAT;
    image_view_create_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    image_view_create_info.subresourceRange.levelCount = 1;
    image_view_create_info.subresourceRange.layerCount = 1;

    res = vkCreateImageView(device->device, &image_view_create_info, NULL, &(target->image_view));
    if (!VULKAN_SUCCESS(res)) {
        status = TEST_VK_IMAGE_VIEW_CREATION_ERROR;
        goto cleanup_image_with_memory;
    }

    VkFramebufferCreateInfo framebuffer_create_info = {0};
    framebuffer_create_info.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    framebuffer_create_info.renderPass = render_pass;
    framebuffer_create_info.attachmentCount = 1;
    framebuffer_create_info.pAttachments = &(target->image_view);
    framebuffer_create_info.width = side;
    framebuffer_create_info.height = side;
    framebuffer_create_info.layers = 1;

    res = vkCreateFramebuffer(device->device, &framebuffer_create_info, NULL, &(target->framebuffer));
    if (!VULKAN_SUCCESS(res)) {
        status = TEST_UNKNOWN_ERROR;
        goto cleanup_image_view;
    }
    return TEST_OK;

cleanup_image_view:
    vkDestroyImageView(device->device, target->image_view, NULL);
    target->image_view = VK_NULL_HANDLE;
cleanup_image_with_memory:
    vkDestroyImage(device->device, target->image, NULL);
    target->image = VK_NULL_HANDLE;
    vkFreeMemory(device->device, target->memory, NULL);
    target->memory = VK_NULL_HANDLE;
    return status;
cleanup_image:
    vkDestroyImage(device->device, target->image, NULL);
    target->image = VK_NULL_HANDLE;
    return status;
}

static test_status _VulkanRatePixelFillCleanTarget(vulkan_rate_pixel_fill_target *target) {
    VkDevice device = target->device->device;
    if (target->framebuffer != VK_NULL_HANDLE) {
        vkDestroyFramebuffer(device, target->framebuffer, NULL);
    }
    if (target->image_view != VK_NULL_HANDLE) {
        vkDestroyImageView(device, target->image_view, NULL);
    }
    if (target->image != VK_NULL_HANDLE) {
        vkDestroyImage(device, target->image, NULL);
    }
    if (target->memory != VK_NULL_HANDLE) {
        vkFreeMemory(device, target->memory, NULL);
    }
    return TEST_OK;
}

static test_status _VulkanRatePixelFillTransitionTarget(vulkan_rate_pixel_fill_target *target, vulkan_command_sequence *command_sequence) {
    test_status status = VulkanCommandBufferStart(command_sequence);
    TEST_RETFAIL(status);

    VkImageMemoryBarrier image_barrier = {0};
    image_barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    image_barrier.srcAccessMask = 0;
    image_barrier.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    image_barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    image_barrier.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    image_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    image_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    image_barrier.image = target->image;
    image_barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    image_barrier.subresourceRange.levelCount = 1;
    image_barrier.subresourceRange.layerCount = 1;

    vkCmdPipelineBarrier(command_sequence->command_buffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 0, 0, NULL, 0, NULL, 1, &image_barrier);

    status = VulkanCommandBufferEnd(command_sequence);
    if (!TEST_SUCCESS(status)) {
        goto cleanup_command_sequence;
    }
    status = VulkanCommandBufferSubmit(command_sequence);
    if (!TEST_SUCCESS(status)) {
        goto cleanup_command_sequence;
    }
    status = VulkanCommandBufferWait(command_sequence, VULKAN_COMMAND_SEQUENCE_WAIT_INFINITE);

cleanup_command_sequence:
    VulkanCommandBufferReset(command_sequence);
    return status;
}

static test_status _VulkanRatePixelFillRunSide(vulkan_device *device, vulkan_rate_pixel_fill_pipeline *pipeline, vulkan_command_sequence *command_sequence, uint32_t side, uint64_t *top_result, uint64_t *top_instances) {
    vulkan_rate_pixel_fill_target target;
    test_status status = _VulkanRatePixelFillCreateTarget(device, pipeline->render_pass, side, &target);
    TEST_RETFAIL(status);

    status = _VulkanRatePixelFillTransitionTarget(&target, command_sequence);
    if (!TEST_SUCCESS(status)) {
        goto cleanup_target;
    }

    INFO("Warming up render target side %lu...\n", side);
    for (uint32_t i = 0; i < VULKAN_RATE_PIXEL_FILL_WARMUP_RUNS; i++) {
        status = _VulkanRatePixelFillExecute(1, side, NULL, NULL, pipeline, &target, command_sequence);
        if (!TEST_SUCCESS(status)) {
            goto cleanup_target;
        }
    }
    INFO("Warmup finished for render target side %lu\n", side);

    *top_result = 0;
    *top_instances = 0;
    uint32_t instance_count = 1;

    while (true) {
        uint64_t result = 0;
        uint64_t time_taken = 0;
        status = _VulkanRatePixelFillExecute(instance_count, side, &result, &time_taken, pipeline, &target, command_sequence);
        if (!TEST_SUCCESS(status)) {
            goto cleanup_target;
        }
        if (result > *top_result) {
            *top_result = result;
            *top_instances = instance_count;
        }
        if (time_taken >= VULKAN_RATE_PIXEL_FILL_TARGET_TIME_US || instance_count >= VULKAN_RATE_PIXEL_FILL_MAX_INSTANCES) {
            break;
        }
        if (instance_count > VULKAN_RATE_PIXEL_FILL_MAX_INSTANCES / 2) {
            instance_count = VULKAN_RATE_PIXEL_FILL_MAX_INSTANCES;
        } else {
            instance_count *= 2;
        }
    }

cleanup_target:
    _VulkanRatePixelFillCleanTarget(&target);
    return status;
}

static test_status _VulkanRatePixelFillExecute(uint32_t instance_count, uint32_t side, uint64_t *result, uint64_t *time_taken, vulkan_rate_pixel_fill_pipeline *pipeline, vulkan_rate_pixel_fill_target *target, vulkan_command_sequence *command_sequence) {
    test_status status = VulkanCommandBufferStart(command_sequence);
    if (!TEST_SUCCESS(status)) {
        goto error;
    }

    VkRenderPassBeginInfo render_pass_begin_info = {0};
    render_pass_begin_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    render_pass_begin_info.renderPass = pipeline->render_pass;
    render_pass_begin_info.framebuffer = target->framebuffer;
    render_pass_begin_info.renderArea.extent.width = side;
    render_pass_begin_info.renderArea.extent.height = side;

    vkCmdBeginRenderPass(command_sequence->command_buffer, &render_pass_begin_info, VK_SUBPASS_CONTENTS_INLINE);

    VkViewport viewport = {0};
    viewport.width = (float)side;
    viewport.height = (float)side;
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;
    vkCmdSetViewport(command_sequence->command_buffer, 0, 1, &viewport);

    VkRect2D scissor = {0};
    scissor.extent.width = side;
    scissor.extent.height = side;
    vkCmdSetScissor(command_sequence->command_buffer, 0, 1, &scissor);

    vkCmdBindPipeline(command_sequence->command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline->pipeline);
    vkCmdDraw(command_sequence->command_buffer, 3, instance_count, 0, 0);
    vkCmdEndRenderPass(command_sequence->command_buffer);

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
        INFO("Instance count %lu render target side %lu took %.3fms (rate: N/A)\n", instance_count, side, time / 1000.0f);
        if (result != NULL) {
            *result = 0;
        }
    } else {
        uint64_t total_pixels = (uint64_t)side * (uint64_t)side * (uint64_t)instance_count;
        uint64_t throughput_per_second = (total_pixels * 1000000) / time;
        helper_unit_pair unit_conversion;
        HelperConvertUnitsPlain1000(throughput_per_second, &unit_conversion);
        INFO("Instance count %lu render target side %lu took %.3fms (rate: %.3f %sPixel/s)\n", instance_count, side, time / 1000.0f, unit_conversion.value, unit_conversion.units);
        if (result != NULL) {
            *result = throughput_per_second;
        }
    }

cleanup_command_sequence:
    VulkanCommandBufferReset(command_sequence);
error:
    return status;
}

static test_status _VulkanRatePixelFillFindMemoryType(vulkan_physical_device *physical_device, uint32_t type_filter, VkMemoryPropertyFlags properties, uint32_t *memory_type_index) {
    VkPhysicalDeviceMemoryProperties *memory_properties = &(physical_device->physical_memory_properties.memoryProperties);
    for (uint32_t i = 0; i < memory_properties->memoryTypeCount; i++) {
        if ((type_filter & (1u << i)) != 0 && (memory_properties->memoryTypes[i].propertyFlags & properties) == properties) {
            *memory_type_index = i;
            return TEST_OK;
        }
    }
    return TEST_VK_SUITABLE_ALLOCATION_NOT_FOUND;
}
