cmake_minimum_required(VERSION 3.28)
project(StellarForge VERSION 2.0.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

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

if(WIN32)
    target_compile_definitions(StellarForge PRIVATE VK_USE_PLATFORM_WIN32_KHR)
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
    ${CMAKE_DL_LIBS}
)

option(STELLAR_BUILD_TESTS "Build CPU state and numerical contract tests" ON)
if(STELLAR_BUILD_TESTS)
    enable_testing()
    add_executable(StellarEclipseTests tests/eclipse_tests.cpp)
    target_include_directories(StellarEclipseTests PRIVATE src)
    target_link_libraries(StellarEclipseTests PRIVATE glm::glm)
    if(MINGW)
        target_link_options(StellarEclipseTests PRIVATE -static-libgcc -static-libstdc++ -Wl,-Bstatic -lstdc++ -lpthread -Wl,-Bdynamic)
    endif()
    add_test(NAME eclipse_contracts COMMAND StellarEclipseTests)
    add_executable(StellarBenchmarkTests tests/benchmark_tests.cpp src/benchmark/benchmark.cpp src/core/path_utils.cpp)
    target_include_directories(StellarBenchmarkTests PRIVATE src)
    target_compile_definitions(StellarBenchmarkTests PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
    target_link_libraries(StellarBenchmarkTests PRIVATE nlohmann_json::nlohmann_json)
    if(MINGW)
        target_link_options(StellarBenchmarkTests PRIVATE -static-libgcc -static-libstdc++ -Wl,-Bstatic -lstdc++ -lpthread -Wl,-Bdynamic)
    endif()
    add_test(NAME benchmark_contracts COMMAND StellarBenchmarkTests "${CMAKE_CURRENT_SOURCE_DIR}/scratch/benchmark-contracts")
    find_package(Python3 QUIET COMPONENTS Interpreter)
    if(Python3_Interpreter_FOUND)
        add_test(NAME benchmark_comparison COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/benchmark_comparison_tests.py")
    endif()
    add_executable(StellarStateTests
        tests/state_tests.cpp
        src/ephemeris/ias15.cpp
        src/ephemeris/reference_hierarchy.cpp
        src/ephemeris/jacobi.cpp
        src/ephemeris/simulation.cpp
        src/ephemeris/spice_ephemeris.cpp
        src/ephemeris/system_manager.cpp
        src/physics/star_calc.cpp
        src/core/path_utils.cpp
        src/core/time_utils.cpp
    )
    target_include_directories(StellarStateTests PRIVATE src)
    target_compile_definitions(StellarStateTests PRIVATE
        GLM_FORCE_RADIANS GLM_FORCE_DEPTH_ZERO_TO_ONE GLM_ENABLE_EXPERIMENTAL NOMINMAX WIN32_LEAN_AND_MEAN
    )
    target_link_libraries(StellarStateTests PRIVATE glm::glm nlohmann_json::nlohmann_json ${CMAKE_DL_LIBS})
    if(MINGW)
        target_link_options(StellarStateTests PRIVATE -static-libgcc -static-libstdc++ -Wl,-Bstatic -lstdc++ -lpthread -Wl,-Bdynamic)
    endif()
    add_test(NAME state_contracts COMMAND StellarStateTests "${CMAKE_CURRENT_SOURCE_DIR}/data/systems")
    get_target_property(STELLAR_RUNNER_TEST_SOURCES StellarStateTests SOURCES)
    list(FILTER STELLAR_RUNNER_TEST_SOURCES EXCLUDE REGEX "tests/state_tests.cpp$")
    add_executable(StellarSimulationRunnerTests tests/simulation_runner_tests.cpp
        src/ephemeris/simulation_runner.cpp ${STELLAR_RUNNER_TEST_SOURCES})
    target_include_directories(StellarSimulationRunnerTests PRIVATE src)
    target_compile_definitions(StellarSimulationRunnerTests PRIVATE
        $<TARGET_PROPERTY:StellarStateTests,COMPILE_DEFINITIONS>)
    target_link_libraries(StellarSimulationRunnerTests PRIVATE $<TARGET_PROPERTY:StellarStateTests,LINK_LIBRARIES>)
    target_link_options(StellarSimulationRunnerTests PRIVATE $<TARGET_PROPERTY:StellarStateTests,LINK_OPTIONS>)
    add_test(NAME simulation_worker_contracts COMMAND StellarSimulationRunnerTests)
    set_tests_properties(simulation_worker_contracts PROPERTIES TIMEOUT 60)
    add_executable(StellarOrbitTests
        tests/orbit_tests.cpp
        src/ephemeris/ias15.cpp
        src/ephemeris/reference_hierarchy.cpp
        src/ephemeris/jacobi.cpp
        src/ephemeris/orbit_curves.cpp
        src/ephemeris/simulation.cpp
        src/ephemeris/spice_ephemeris.cpp
        src/ephemeris/system_manager.cpp
        src/physics/star_calc.cpp
        src/core/path_utils.cpp
    )
    target_include_directories(StellarOrbitTests PRIVATE src)
    target_compile_definitions(StellarOrbitTests PRIVATE
        GLM_FORCE_RADIANS GLM_FORCE_DEPTH_ZERO_TO_ONE GLM_ENABLE_EXPERIMENTAL NOMINMAX WIN32_LEAN_AND_MEAN
    )
    target_link_libraries(StellarOrbitTests PRIVATE glm::glm nlohmann_json::nlohmann_json ${CMAKE_DL_LIBS})
    if(MINGW)
        target_link_options(StellarOrbitTests PRIVATE -static-libgcc -static-libstdc++ -Wl,-Bstatic -lstdc++ -lpthread -Wl,-Bdynamic)
    endif()
    add_test(NAME orbit_contracts COMMAND StellarOrbitTests "${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures" "${CMAKE_CURRENT_SOURCE_DIR}/data/systems")
    add_executable(StellarHierarchyTests
        tests/hierarchy_tests.cpp
        src/ephemeris/ias15.cpp
        src/ephemeris/jacobi.cpp
        src/ephemeris/reference_hierarchy.cpp
        src/ephemeris/orbit_curves.cpp
        src/ephemeris/simulation.cpp
        src/ephemeris/spice_ephemeris.cpp
        src/ephemeris/system_manager.cpp
        src/physics/star_calc.cpp
        src/core/path_utils.cpp
    )
    target_include_directories(StellarHierarchyTests PRIVATE src)
    target_compile_definitions(StellarHierarchyTests PRIVATE
        GLM_FORCE_RADIANS GLM_FORCE_DEPTH_ZERO_TO_ONE GLM_ENABLE_EXPERIMENTAL NOMINMAX WIN32_LEAN_AND_MEAN
    )
    target_link_libraries(StellarHierarchyTests PRIVATE glm::glm nlohmann_json::nlohmann_json ${CMAKE_DL_LIBS})
    if(MINGW)
        target_link_options(StellarHierarchyTests PRIVATE -static-libgcc -static-libstdc++ -Wl,-Bstatic -lstdc++ -lpthread -Wl,-Bdynamic)
    endif()
    add_test(NAME hierarchy_contracts COMMAND StellarHierarchyTests)
    add_executable(StellarStarCalcTests
        tests/star_calc_tests.cpp
        src/physics/star_calc.cpp
    )
    target_include_directories(StellarStarCalcTests PRIVATE src)
    target_compile_definitions(StellarStarCalcTests PRIVATE
        GLM_FORCE_RADIANS GLM_FORCE_DEPTH_ZERO_TO_ONE GLM_ENABLE_EXPERIMENTAL NOMINMAX WIN32_LEAN_AND_MEAN
    )
    target_link_libraries(StellarStarCalcTests PRIVATE glm::glm nlohmann_json::nlohmann_json)
    if(MINGW)
        target_link_options(StellarStarCalcTests PRIVATE -static-libgcc -static-libstdc++ -Wl,-Bstatic -lstdc++ -lpthread -Wl,-Bdynamic)
    endif()
    add_test(NAME star_calc_contracts COMMAND StellarStarCalcTests)
    option(STELLAR_TEST_EPHEMERIS "Test against locally installed CSPICE kernels" OFF)
    if(STELLAR_TEST_EPHEMERIS)
        target_compile_definitions(StellarStateTests PRIVATE STELLAR_TEST_EPHEMERIS)
        add_test(NAME ephemeris_contracts COMMAND StellarOrbitTests "${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures" "${CMAKE_CURRENT_SOURCE_DIR}/data/systems" --spice)
    endif()
endif()

option(STELLAR_BUILD_RENDER_TESTS "Build GPU rendering contract tests (requires Vulkan and a window system)" OFF)
if(STELLAR_BUILD_RENDER_TESTS)
    enable_testing()
    add_test(NAME simulation_tracking_rendering_contracts COMMAND StellarForge --self-test)
    set_tests_properties(simulation_tracking_rendering_contracts PROPERTIES TIMEOUT 30)
    add_executable(StellarRenderingTests
        tests/rendering_tests.cpp
        src/ephemeris/ias15.cpp
        src/ephemeris/reference_hierarchy.cpp
        src/ephemeris/jacobi.cpp
        src/core/camera.cpp
        src/core/path_utils.cpp
        src/ephemeris/simulation.cpp
        src/ephemeris/spice_ephemeris.cpp
        src/ephemeris/system_manager.cpp
        src/physics/star_calc.cpp
        src/rendering/celestial_renderer.cpp
        src/rendering/star_catalog.cpp
        src/rhi/shader_cache.cpp
        src/rhi/vk_context.cpp
    )
    target_include_directories(StellarRenderingTests PRIVATE $<TARGET_PROPERTY:StellarForge,INCLUDE_DIRECTORIES>)
    target_compile_definitions(StellarRenderingTests PRIVATE $<TARGET_PROPERTY:StellarForge,COMPILE_DEFINITIONS>)
    target_link_libraries(StellarRenderingTests PRIVATE $<TARGET_PROPERTY:StellarForge,LINK_LIBRARIES>)
    target_link_options(StellarRenderingTests PRIVATE $<TARGET_PROPERTY:StellarForge,LINK_OPTIONS>)
    add_test(NAME rendering_contracts COMMAND StellarRenderingTests)
    add_executable(StellarOrbitRenderingTests
        tests/orbit_rendering_tests.cpp
        src/ephemeris/ias15.cpp
        src/ephemeris/reference_hierarchy.cpp
        src/ephemeris/jacobi.cpp
        src/core/camera.cpp
        src/core/path_utils.cpp
        src/ephemeris/orbit_curves.cpp
        src/ephemeris/simulation.cpp
        src/ephemeris/spice_ephemeris.cpp
        src/ephemeris/system_manager.cpp
        src/physics/star_calc.cpp
        src/rendering/orbit_renderer.cpp
        src/rhi/shader_cache.cpp
        src/rhi/vk_context.cpp
    )
    target_include_directories(StellarOrbitRenderingTests PRIVATE $<TARGET_PROPERTY:StellarForge,INCLUDE_DIRECTORIES>)
    target_compile_definitions(StellarOrbitRenderingTests PRIVATE $<TARGET_PROPERTY:StellarForge,COMPILE_DEFINITIONS>)
    target_link_libraries(StellarOrbitRenderingTests PRIVATE $<TARGET_PROPERTY:StellarForge,LINK_LIBRARIES>)
    target_link_options(StellarOrbitRenderingTests PRIVATE $<TARGET_PROPERTY:StellarForge,LINK_OPTIONS>)
    add_test(NAME orbit_rendering_contracts COMMAND StellarOrbitRenderingTests)
endif()
