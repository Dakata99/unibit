# Fetch Taywee/args (CL argument parser)
include(FetchContent)

FetchContent_Declare(
    args
    GIT_REPOSITORY https://github.com/Taywee/args.git
    GIT_TAG 6.4.7   # stable release tag
)

FetchContent_MakeAvailable(args)
