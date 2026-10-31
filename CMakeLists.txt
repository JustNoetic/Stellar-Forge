cmake_minimum_required(VERSION 3.28)
project(StellarForge VERSION 2.0.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

include(FetchContent)

FetchContent_Declare(
    vulkan_headers
    GIT_REPOSITORY https://github.com/KhronosGroup/Vulkan-Headers.git
    GIT_TAG v1.3.296
    GIT_SHALLOW TRUE
)

FetchContent_Declare(
    volk
    GIT_REPOSITORY https://github.com/zeux/volk.git
    GIT_TAG vulkan-sdk-1.3.296.0
    GIT_SHALLOW TRUE
)

FetchContent_Declare(
    vma
    GIT_REPOSITORY https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator.git
    GIT_TAG v3.1.0
    GIT_SHALLOW TRUE
)

set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
    glfw
    GIT_REPOSITORY https://github.com/glfw/glfw.git
    GIT_TAG 3.4
    GIT_SHALLOW TRUE
)

FetchContent_Declare(
    glm
    GIT_REPOSITORY https://github.com/g-truc/glm.git
    GIT_TAG 1.0.1
    GIT_SHALLOW TRUE
)

FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.3
    GIT_SHALLOW TRUE
)

FetchContent_Declare(
    imgui
    GIT_REPOSITORY https://github.com/ocornut/imgui.git
    GIT_TAG v1.91.8-docking
    GIT_SHALLOW TRUE
)

set(ENABLE_GLSLANG_BINARIES OFF CACHE BOOL "" FORCE)
set(ENABLE_SPVREMAPPER OFF CACHE BOOL "" FORCE)
set(ENABLE_HLSL OFF CACHE BOOL "" FORCE)
set(ENABLE_OPT OFF CACHE BOOL "" FORCE)
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(GLSLANG_TESTS OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
    glslang
    GIT_REPOSITORY https://github.com/KhronosGroup/glslang.git
    GIT_TAG 15.1.0
    GIT_SHALLOW TRUE
)

FetchContent_MakeAvailable(vulkan_headers volk vma glfw glm nlohmann_json imgui glslang)
target_link_libraries(volk PUBLIC Vulkan::Headers)

add_library(imgui_lib STATIC
    ${imgui_SOURCE_DIR}/imgui.cpp
    ${imgui_SOURCE_DIR}/imgui_draw.cpp
    ${imgui_SOURCE_DIR}/imgui_tables.cpp
    ${imgui_SOURCE_DIR}/imgui_widgets.cpp
    ${imgui_SOURCE_DIR}/imgui_demo.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_vulkan.cpp
)
target_include_directories(imgui_lib PUBLIC
    ${imgui_SOURCE_DIR}
    ${imgui_SOURCE_DIR}/backends
)
target_compile_definitions(imgui_lib PUBLIC
    VK_NO_PROTOTYPES
    IMGUI_IMPL_VULKAN_USE_VOLK
)
target_link_libraries(imgui_lib PUBLIC glfw volk_headers Vulkan::Headers)

file(GLOB_RECURSE STELLAR_SOURCES CONFIGURE_DEPENDS
    "src/*.cpp"
)

add_executable(StellarForge ${STELLAR_SOURCES})

execute_process(COMMAND git rev-parse HEAD WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    OUTPUT_VARIABLE STELLAR_GIT_COMMIT OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
execute_process(COMMAND git diff --quiet HEAD WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    RESULT_VARIABLE STELLAR_GIT_DIFF ERROR_QUIET)
if(NOT STELLAR_GIT_COMMIT)
    set(STELLAR_GIT_COMMIT "unavailable")
endif()
if(STELLAR_GIT_DIFF STREQUAL "0")
    set(STELLAR_GIT_DIRTY false)
else()
    set(STELLAR_GIT_DIRTY true)
endif()
configure_file(src/benchmark/build_info.hpp.in generated/benchmark/build_info.hpp @ONLY)

target_include_directories(StellarForge PRIVATE
    src
    "${CMAKE_CURRENT_BINARY_DIR}/generated"
    ${vma_SOURCE_DIR}/include
)

target_compile_definitions(StellarForge PRIVATE
    VK_NO_PROTOTYPES
    GLM_FORCE_RADIANS
    GLM_FORCE_DEPTH_ZERO_TO_ONE
    GLM_ENABLE_EXPERIMENTAL
    NOMINMAX
    WIN32_LEAN_AND_MEAN
)

set(STELLAR_PLATFORM_LIBS ${CMAKE_DL_LIBS})
if(WIN32)
    target_compile_definitions(StellarForge PRIVATE VK_USE_PLATFORM_WIN32_KHR)
    list(APPEND STELLAR_PLATFORM_LIBS winhttp comdlg32 ole32 windowscodecs uuid)
endif()

if(MINGW)
    target_link_options(StellarForge PRIVATE -static-libgcc -static-libstdc++ -Wl,-Bstatic -lstdc++ -lpthread -Wl,-Bdynamic)
endif()

target_link_libraries(StellarForge PRIVATE
    Vulkan::Headers
    volk_headers
    glfw
    glm::glm
    nlohmann_json::nlohmann_json
    imgui_lib
    glslang
    glslang-default-resource-limits
    SPIRV
    ${STELLAR_PLATFORM_LIBS}
)

# --- Testing ---
option(STELLAR_BUILD_TESTS "Build CPU state and numerical contract tests" ON)
option(STELLAR_BUILD_RENDER_TESTS "Build GPU rendering contract tests (requires Vulkan and a window system)" OFF)
option(STELLAR_TEST_EPHEMERIS "Test against locally installed CSPICE kernels" OFF)

if((STELLAR_BUILD_TESTS OR STELLAR_BUILD_RENDER_TESTS) AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/CMakeLists.txt")
    enable_testing()
    add_subdirectory(tests)
endif()

# --- Deployment / Packaging Staging ---
install(TARGETS StellarForge RUNTIME DESTINATION .)
install(PROGRAMS "${CMAKE_CURRENT_SOURCE_DIR}/bake_textures.bat" DESTINATION .)
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/data" DESTINATION .
    PATTERN "settings.json" EXCLUDE
    PATTERN "settings.json.tmp" EXCLUDE
)
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/shaders" DESTINATION .
    PATTERN "tests" EXCLUDE
)
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/scripts" DESTINATION .
    PATTERN "__pycache__" EXCLUDE
    PATTERN "*.pyc" EXCLUDE
)
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/textures" DESTINATION .)
