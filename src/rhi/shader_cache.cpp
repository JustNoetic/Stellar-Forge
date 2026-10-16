#include "rhi/shader_cache.hpp"
#include "core/path_utils.hpp"

#include <glslang/Public/ResourceLimits.h>
#include <glslang/Public/ShaderLang.h>
#include <SPIRV/GlslangToSpv.h>

#include <format>
#include <sstream>
#include <unordered_set>

namespace sf::rhi {

namespace {

uint64_t fnv1a_64(std::string_view data, uint64_t seed = 14695981039346656037ULL) noexcept {
    uint64_t hash = seed;
    for (unsigned char c : data) {
        hash ^= static_cast<uint64_t>(c);
        hash *= 1099511628211ULL;
    }
    return hash;
}

EShLanguage to_glslang_stage(VkShaderStageFlagBits stage) {
    switch (stage) {
        case VK_SHADER_STAGE_VERTEX_BIT: return EShLangVertex;
        case VK_SHADER_STAGE_GEOMETRY_BIT: return EShLangGeometry;
        case VK_SHADER_STAGE_FRAGMENT_BIT: return EShLangFragment;
        case VK_SHADER_STAGE_COMPUTE_BIT: return EShLangCompute;
        default: return EShLangVertex;
    }
}

std::string resolve_includes_recursive(const std::filesystem::path& file_path,
                                       const std::filesystem::path& shaders_root,
                                       std::unordered_set<std::string>& visited) {
    auto canonical_str = std::filesystem::weakly_canonical(file_path).string();
    if (visited.contains(canonical_str)) {
        return {};
    }
    visited.insert(canonical_str);

    auto content = sf::paths::read_text_file(file_path);
    if (!content) {
        sf::log::fatal("Failed to read shader file: {}", file_path.string());
    }

    std::istringstream iss(*content);
    std::ostringstream oss;
    std::string line;

    while (std::getline(iss, line)) {
        std::string trimmed = line;
        auto first = trimmed.find_first_not_of(" \t\r");
        if (first != std::string::npos && trimmed.compare(first, 8, "#include") == 0) {
            auto q1 = trimmed.find('"', first + 8);
            auto q2 = (q1 != std::string::npos) ? trimmed.find('"', q1 + 1) : std::string::npos;
            if (q1 != std::string::npos && q2 != std::string::npos) {
                std::string inc_rel = trimmed.substr(q1 + 1, q2 - q1 - 1);
                std::filesystem::path candidate = file_path.parent_path() / inc_rel;
                if (!std::filesystem::exists(candidate)) {
                    candidate = shaders_root / inc_rel;
                }
                oss << resolve_includes_recursive(candidate, shaders_root, visited) << "\n";
                continue;
            }
        }
        oss << line << "\n";
    }
    return oss.str();
}

VkDescriptorSetLayout create_push_set_layout(VkDevice device, std::span<const DescriptorLayoutBinding> bindings) {
    std::vector<VkDescriptorSetLayoutBinding> vk_bindings(bindings.size());
    for (size_t i = 0; i < bindings.size(); ++i) {
        vk_bindings[i].binding = bindings[i].binding;
        vk_bindings[i].descriptorType = bindings[i].type;
        vk_bindings[i].descriptorCount = bindings[i].count;
        vk_bindings[i].stageFlags = bindings[i].stages;
    }

    VkDescriptorSetLayoutCreateInfo dsl_ci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dsl_ci.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR;
    dsl_ci.bindingCount = static_cast<uint32_t>(vk_bindings.size());
    dsl_ci.pBindings = vk_bindings.data();

    VkDescriptorSetLayout set_layout = VK_NULL_HANDLE;
    VK_CHECK(vkCreateDescriptorSetLayout(device, &dsl_ci, nullptr, &set_layout));
    return set_layout;
}

VkPipelineLayout create_pipeline_layout(VkDevice device, VkDescriptorSetLayout set_layout,
                                        uint32_t push_constant_size, VkShaderStageFlags push_stages) {
    VkPushConstantRange pc_range{};
    pc_range.stageFlags = push_stages;
    pc_range.offset = 0;
    pc_range.size = push_constant_size;

    VkPipelineLayoutCreateInfo pl_ci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    if (set_layout != VK_NULL_HANDLE) {
        pl_ci.setLayoutCount = 1;
        pl_ci.pSetLayouts = &set_layout;
    }
    if (push_constant_size > 0) {
        pl_ci.pushConstantRangeCount = 1;
        pl_ci.pPushConstantRanges = &pc_range;
    }

    VkPipelineLayout layout = VK_NULL_HANDLE;
    VK_CHECK(vkCreatePipelineLayout(device, &pl_ci, nullptr, &layout));
    return layout;
}

} // namespace

void ShaderCache::init() {
    glslang::InitializeProcess();
}

void ShaderCache::shutdown() {
    glslang::FinalizeProcess();
}

std::string ShaderCache::load_and_preprocess(const std::filesystem::path& shader_path,
                                             std::span<const std::pair<std::string, std::string>> macros) {
    std::filesystem::path shaders_root = sf::paths::bundled_path("shaders");
    std::unordered_set<std::string> visited;
    std::string source = resolve_includes_recursive(shader_path, shaders_root, visited);

    if (macros.empty()) {
        return source;
    }

    std::ostringstream macro_block;
    for (const auto& [k, v] : macros) {
        if (v.empty()) {
            macro_block << "#define " << k << "\n";
        } else {
            macro_block << "#define " << k << " " << v << "\n";
        }
    }

    auto ver_pos = source.find("#version");
    if (ver_pos != std::string::npos) {
        auto line_end = source.find('\n', ver_pos);
        if (line_end != std::string::npos) {
            source.insert(line_end + 1, macro_block.str());
            return source;
        }
    }
    return macro_block.str() + source;
}

std::vector<uint32_t> ShaderCache::compile_to_spirv(std::string_view relative_path,
                                                    VkShaderStageFlagBits stage,
                                                    std::span<const std::pair<std::string, std::string>> macros) {
    std::filesystem::path full_path = sf::paths::bundled_path("shaders", relative_path);
    std::string source = load_and_preprocess(full_path, macros);

    uint64_t hash = fnv1a_64(source);
    hash = fnv1a_64(std::string_view(reinterpret_cast<const char*>(&stage), sizeof(stage)), hash);

    std::filesystem::path cache_file = sf::paths::external_path("cache", "shaders", std::format("{:016x}.spv", hash));
    if (auto cached = sf::paths::read_binary_file(cache_file); cached && cached->size() % 4 == 0 && !cached->empty()) {
        std::vector<uint32_t> spv(cached->size() / 4);
        std::memcpy(spv.data(), cached->data(), cached->size());
        return spv;
    }

    EShLanguage lang = to_glslang_stage(stage);
    glslang::TShader shader(lang);
    const char* src_ptr = source.c_str();
    shader.setStrings(&src_ptr, 1);
    shader.setEnvInput(glslang::EShSourceGlsl, lang, glslang::EShClientVulkan, 460);
    shader.setEnvClient(glslang::EShClientVulkan, glslang::EShTargetVulkan_1_3);
    shader.setEnvTarget(glslang::EShTargetSpv, glslang::EShTargetSpv_1_6);

    EShMessages messages = static_cast<EShMessages>(EShMsgSpvRules | EShMsgVulkanRules);
    if (!shader.parse(GetDefaultResources(), 460, false, messages)) {
        sf::log::fatal("Shader parse failed ({}):\n{}", relative_path, shader.getInfoLog());
    }

    glslang::TProgram program;
    program.addShader(&shader);
    if (!program.link(messages)) {
        sf::log::fatal("Shader link failed ({}):\n{}", relative_path, program.getInfoLog());
    }

    std::vector<uint32_t> spirv;
    glslang::SpvOptions spv_opts{};
    spv_opts.disableOptimizer = false;
    glslang::GlslangToSpv(*program.getIntermediate(lang), spirv, &spv_opts);

    sf::paths::write_binary_file(cache_file, spirv.data(), spirv.size() * sizeof(uint32_t));
    return spirv;
}

VkShaderModule ShaderCache::create_shader_module(VkDevice device,
                                                 std::string_view relative_path,
                                                 VkShaderStageFlagBits stage,
                                                 std::span<const std::pair<std::string, std::string>> macros) {
    std::vector<uint32_t> spirv = compile_to_spirv(relative_path, stage, macros);
    VkShaderModuleCreateInfo sm_ci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    sm_ci.codeSize = spirv.size() * sizeof(uint32_t);
    sm_ci.pCode = spirv.data();

    VkShaderModule mod = VK_NULL_HANDLE;
    VK_CHECK(vkCreateShaderModule(device, &sm_ci, nullptr, &mod));
    return mod;
}

Pipeline ShaderCache::create_compute_pipeline(VkContext& ctx,
                                              std::string_view comp_path,
                                              std::span<const DescriptorLayoutBinding> bindings,
                                              uint32_t push_constant_size,
                                              std::span<const std::pair<std::string, std::string>> macros) {
    Pipeline pipe{};
    pipe.bind_point = VK_PIPELINE_BIND_POINT_COMPUTE;
    pipe.push_constant_stages = VK_SHADER_STAGE_COMPUTE_BIT;
    pipe.set_layout = create_push_set_layout(ctx.device(), bindings);
    pipe.layout = create_pipeline_layout(ctx.device(), pipe.set_layout, push_constant_size, pipe.push_constant_stages);

    VkShaderModule mod = create_shader_module(ctx.device(), comp_path, VK_SHADER_STAGE_COMPUTE_BIT, macros);

    VkPipelineShaderStageCreateInfo stage_ci{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stage_ci.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stage_ci.module = mod;
    stage_ci.pName = "main";

    VkComputePipelineCreateInfo cp_ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cp_ci.stage = stage_ci;
    cp_ci.layout = pipe.layout;

    VK_CHECK(vkCreateComputePipelines(ctx.device(), VK_NULL_HANDLE, 1, &cp_ci, nullptr, &pipe.handle));
    vkDestroyShaderModule(ctx.device(), mod, nullptr);
    return pipe;
}

Pipeline ShaderCache::create_graphics_pipeline(VkContext& ctx, const GraphicsPipelineDesc& desc) {
    Pipeline pipe{};
    pipe.bind_point = VK_PIPELINE_BIND_POINT_GRAPHICS;
    pipe.push_constant_stages = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    if (!desc.geom_path.empty()) {
        pipe.push_constant_stages |= VK_SHADER_STAGE_GEOMETRY_BIT;
    }

    pipe.set_layout = create_push_set_layout(ctx.device(), desc.bindings);
    pipe.layout = create_pipeline_layout(ctx.device(), pipe.set_layout, desc.push_constant_size, pipe.push_constant_stages);

    std::vector<VkPipelineShaderStageCreateInfo> stages;
    std::vector<VkShaderModule> modules;

    auto add_stage = [&](const std::string& path, VkShaderStageFlagBits stage) {
        if (path.empty()) return;
        VkShaderModule m = create_shader_module(ctx.device(), path, stage, desc.macros);
        modules.push_back(m);
        VkPipelineShaderStageCreateInfo s_ci{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        s_ci.stage = stage;
        s_ci.module = m;
        s_ci.pName = "main";
        stages.push_back(s_ci);
    };

    add_stage(desc.vert_path, VK_SHADER_STAGE_VERTEX_BIT);
    add_stage(desc.geom_path, VK_SHADER_STAGE_GEOMETRY_BIT);
    add_stage(desc.frag_path, VK_SHADER_STAGE_FRAGMENT_BIT);

    std::vector<VkVertexInputBindingDescription> v_bindings(desc.vertex_bindings.size());
    for (size_t i = 0; i < desc.vertex_bindings.size(); ++i) {
        v_bindings[i] = {desc.vertex_bindings[i].binding, desc.vertex_bindings[i].stride, desc.vertex_bindings[i].input_rate};
    }
    std::vector<VkVertexInputAttributeDescription> v_attrs(desc.vertex_attributes.size());
    for (size_t i = 0; i < desc.vertex_attributes.size(); ++i) {
        v_attrs[i] = {desc.vertex_attributes[i].location, desc.vertex_attributes[i].binding,
                      desc.vertex_attributes[i].format, desc.vertex_attributes[i].offset};
    }

    VkPipelineVertexInputStateCreateInfo vi_ci{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    vi_ci.vertexBindingDescriptionCount = static_cast<uint32_t>(v_bindings.size());
    vi_ci.pVertexBindingDescriptions = v_bindings.data();
    vi_ci.vertexAttributeDescriptionCount = static_cast<uint32_t>(v_attrs.size());
    vi_ci.pVertexAttributeDescriptions = v_attrs.data();

    VkPipelineInputAssemblyStateCreateInfo ia_ci{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    ia_ci.topology = desc.topology;

    VkPipelineViewportStateCreateInfo vp_ci{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    vp_ci.viewportCount = 1;
    vp_ci.scissorCount = 1;

    VkPipelineRasterizationStateCreateInfo rs_ci{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    rs_ci.polygonMode = desc.polygon_mode;
    rs_ci.cullMode = desc.cull_mode;
    rs_ci.frontFace = desc.front_face;
    rs_ci.lineWidth = 1.0f;
    VkPipelineRasterizationLineStateCreateInfoKHR line_state{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_LINE_STATE_CREATE_INFO_KHR};
    if (desc.bresenham_lines && ctx.bresenham_lines_enabled()) {
        line_state.lineRasterizationMode = VK_LINE_RASTERIZATION_MODE_BRESENHAM_KHR;
        rs_ci.pNext = &line_state;
    }

    VkPipelineMultisampleStateCreateInfo ms_ci{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    ms_ci.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineDepthStencilStateCreateInfo ds_ci{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    ds_ci.depthTestEnable = desc.depth_test ? VK_TRUE : VK_FALSE;
    ds_ci.depthWriteEnable = desc.depth_write ? VK_TRUE : VK_FALSE;
    ds_ci.depthCompareOp = desc.depth_compare;

    std::vector<VkPipelineColorBlendAttachmentState> blend_atts(desc.color_formats.size());
    for (auto& b : blend_atts) {
        b.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                           VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        if (desc.alpha_blend) {
            b.blendEnable = VK_TRUE;
            b.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
            b.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
            b.colorBlendOp = VK_BLEND_OP_ADD;
            b.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
            b.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
            b.alphaBlendOp = VK_BLEND_OP_ADD;
        } else if (desc.additive_blend) {
            b.blendEnable = VK_TRUE;
            b.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
            b.dstColorBlendFactor = VK_BLEND_FACTOR_ONE;
            b.colorBlendOp = VK_BLEND_OP_ADD;
            b.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
            b.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
            b.alphaBlendOp = VK_BLEND_OP_ADD;
        }
    }

    VkPipelineColorBlendStateCreateInfo cb_ci{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    cb_ci.attachmentCount = static_cast<uint32_t>(blend_atts.size());
    cb_ci.pAttachments = blend_atts.data();

    VkDynamicState dyn_states[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dyn_ci{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    dyn_ci.dynamicStateCount = static_cast<uint32_t>(std::size(dyn_states));
    dyn_ci.pDynamicStates = dyn_states;

    VkPipelineRenderingCreateInfo dyn_render_ci{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
    dyn_render_ci.colorAttachmentCount = static_cast<uint32_t>(desc.color_formats.size());
    dyn_render_ci.pColorAttachmentFormats = desc.color_formats.data();
    dyn_render_ci.depthAttachmentFormat = desc.depth_format;

    VkGraphicsPipelineCreateInfo gp_ci{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    gp_ci.pNext = &dyn_render_ci;
    gp_ci.stageCount = static_cast<uint32_t>(stages.size());
    gp_ci.pStages = stages.data();
    gp_ci.pVertexInputState = &vi_ci;
    gp_ci.pInputAssemblyState = &ia_ci;
    gp_ci.pViewportState = &vp_ci;
    gp_ci.pRasterizationState = &rs_ci;
    gp_ci.pMultisampleState = &ms_ci;
    gp_ci.pDepthStencilState = &ds_ci;
    gp_ci.pColorBlendState = &cb_ci;
    gp_ci.pDynamicState = &dyn_ci;
    gp_ci.layout = pipe.layout;

    VK_CHECK(vkCreateGraphicsPipelines(ctx.device(), VK_NULL_HANDLE, 1, &gp_ci, nullptr, &pipe.handle));

    for (VkShaderModule m : modules) {
        vkDestroyShaderModule(ctx.device(), m, nullptr);
    }
    return pipe;
}

void ShaderCache::destroy_pipeline(VkContext& ctx, Pipeline& pipeline) {
    if (pipeline.handle != VK_NULL_HANDLE) {
        vkDestroyPipeline(ctx.device(), pipeline.handle, nullptr);
    }
    if (pipeline.layout != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(ctx.device(), pipeline.layout, nullptr);
    }
    if (pipeline.set_layout != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(ctx.device(), pipeline.set_layout, nullptr);
    }
    pipeline = {};
}

} // namespace sf::rhi
