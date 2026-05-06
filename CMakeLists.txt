cmake_minimum_required(VERSION 3.20)
project(instanced_tapered_capsules_webgl2 LANGUAGES CXX)

if (NOT EMSCRIPTEN)
    message(FATAL_ERROR "This project targets WebGL2 through Emscripten. Configure it with `emcmake cmake -S . -B build-web`.")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

include(FetchContent)

FetchContent_Declare(
    imgui
    GIT_REPOSITORY https://github.com/ocornut/imgui.git
    GIT_TAG v1.92.7
)
FetchContent_MakeAvailable(imgui)

add_library(imgui STATIC
    ${imgui_SOURCE_DIR}/imgui.cpp
    ${imgui_SOURCE_DIR}/imgui_draw.cpp
    ${imgui_SOURCE_DIR}/imgui_tables.cpp
    ${imgui_SOURCE_DIR}/imgui_widgets.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_opengl3.cpp
)

target_include_directories(imgui PUBLIC
    ${imgui_SOURCE_DIR}
    ${imgui_SOURCE_DIR}/backends
)

target_compile_options(imgui PUBLIC
    -sUSE_SDL=2
)

add_executable(index
    src/main.cpp
)

target_link_libraries(index PRIVATE imgui)

target_compile_options(index PRIVATE
    -sUSE_SDL=2
)

target_link_options(index PRIVATE
    -sUSE_SDL=2
    -sUSE_WEBGL2=1
    -sMIN_WEBGL_VERSION=2
    -sMAX_WEBGL_VERSION=2
    -sFULL_ES3=1
    -sALLOW_MEMORY_GROWTH=1
    -sNO_EXIT_RUNTIME=1
    -sDISABLE_EXCEPTION_CATCHING=1
    --shell-file=${CMAKE_SOURCE_DIR}/web/shell.html
    --preload-file=${imgui_SOURCE_DIR}/misc/fonts/Roboto-Medium.ttf@/fonts/Roboto-Medium.ttf
)

set_target_properties(index PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/web"
    SUFFIX ".html"
)

add_custom_command(TARGET index POST_BUILD
    COMMAND ${CMAKE_COMMAND}
        -DHTML=$<TARGET_FILE:index>
        -DVERSION=20260506-webgl2
        -P ${CMAKE_SOURCE_DIR}/web/cache_bust.cmake
)
