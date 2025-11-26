# spdlog tool for logging

include(FetchContent)

FetchContent_Declare(
    spdlog
    GIT_REPOSITORY https://github.com/gabime/spdlog.git
    GIT_TAG v1.16.0   # use a stable tagged release
)

FetchContent_MakeAvailable(spdlog)
