// Copyright (c) 2021 - 2023, Nemes <nemes@nemez.net>
// SPDX-License-Identifier: MIT

#include "main.h"
#include "logger.h"
#include "runner.h"
#include "build_info.h"

static test_result_output result_format = test_result_csv;
static const char *binary_path = NULL;

static void PrintUsage(const char *program) {
    fprintf(stderr, "vkperf %u.%u.%u.%s%s\n",
            TEST_VER_MAJOR(TEST_TOOL_VERSION),
            TEST_VER_MINOR(TEST_TOOL_VERSION),
            TEST_VER_PATCH(TEST_TOOL_VERSION),
            BUILD_INFO_IDENTIFIER,
            BUILD_INFO_TYPE_SUFFIX);
    fprintf(stderr, "\n");
    fprintf(stderr, "Usage:\n");
    fprintf(stderr, "  %s --test <test id> [--device <device id>] [--format csv|raw|readable]\n", program);
    fprintf(stderr, "  %s --list-tests\n", program);
    fprintf(stderr, "\n");
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -t, --test <id>        Test to run. Example: vk_list, vk_info, vk_rate_fp32_add\n");
    fprintf(stderr, "  -d, --device <id>      Device index. Default: -1 (all matching devices)\n");
    fprintf(stderr, "  -f, --format <format>  csv, raw, or readable. Default: csv\n");
    fprintf(stderr, "  -s, --csv             Shorthand for --format csv\n");
    fprintf(stderr, "  -r, --raw             Shorthand for --format raw\n");
    fprintf(stderr, "  --list-tests          Print available test IDs to stderr\n");
    fprintf(stderr, "  -h, --help            Print this help\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "Available tests:\n");
    RunnerPrintTests();
}

static bool ReadArgumentValue(int argc, const char **argv, int *index, const char **value) {
    if (*index + 1 >= argc) {
        return false;
    }
    *index += 1;
    *value = argv[*index];
    return true;
}

int main(int argc, const char **argv) {
    binary_path = argc > 0 ? argv[0] : "vkperf";

    test_status status = RunnerRegisterTests();
    if (!TEST_SUCCESS(status)) {
        ABORT(status);
        return 1;
    }

    int32_t gpu_identifier = -1;
    const char *test_identifier = NULL;
    bool print_help = false;
    bool list_tests = false;

    for (int i = 1; i < argc; i++) {
        const char *arg = argv[i];
        const char *value = NULL;

        if (strcmp(arg, "--help") == 0 || strcmp(arg, "-h") == 0) {
            print_help = true;
        } else if (strcmp(arg, "--list-tests") == 0) {
            list_tests = true;
        } else if (strcmp(arg, "--csv") == 0 || strcmp(arg, "-s") == 0) {
            result_format = test_result_csv;
        } else if (strcmp(arg, "--raw") == 0 || strcmp(arg, "-r") == 0) {
            result_format = test_result_raw;
        } else if (strcmp(arg, "--device") == 0 || strcmp(arg, "-d") == 0) {
            if (!ReadArgumentValue(argc, argv, &i, &value)) {
                FATAL("Missing value for %s\n", arg);
                RunnerCleanUp();
                return 1;
            }
            gpu_identifier = (int32_t)strtol(value, NULL, 10);
        } else if (strcmp(arg, "--test") == 0 || strcmp(arg, "-t") == 0) {
            if (!ReadArgumentValue(argc, argv, &i, &value)) {
                FATAL("Missing value for %s\n", arg);
                RunnerCleanUp();
                return 1;
            }
            test_identifier = value;
        } else if (strcmp(arg, "--format") == 0 || strcmp(arg, "-f") == 0) {
            if (!ReadArgumentValue(argc, argv, &i, &value)) {
                FATAL("Missing value for %s\n", arg);
                RunnerCleanUp();
                return 1;
            }
            if (strcmp(value, "csv") == 0) {
                result_format = test_result_csv;
            } else if (strcmp(value, "raw") == 0) {
                result_format = test_result_raw;
            } else if (strcmp(value, "readable") == 0) {
                result_format = test_result_readable;
            } else {
                FATAL("Unknown format: %s\n", value);
                RunnerCleanUp();
                return 1;
            }
        } else {
            FATAL("Unknown argument: %s\n", arg);
            RunnerCleanUp();
            return 1;
        }
    }

    if (print_help) {
        PrintUsage(binary_path);
        RunnerCleanUp();
        return 0;
    }
    if (list_tests) {
        RunnerPrintTests();
        RunnerCleanUp();
        return 0;
    }
    if (test_identifier == NULL) {
        ABORT(TEST_NO_TEST_SPECIFIED);
        PrintUsage(binary_path);
        RunnerCleanUp();
        return 1;
    }

    status = RunnerExecuteTests(test_identifier, gpu_identifier);
    RunnerCleanUp();
    if (!TEST_SUCCESS(status)) {
        ABORT(status);
        return 1;
    }
    return 0;
}

test_result_output MainGetTestResultFormat() {
    return result_format;
}

const char *MainGetBinaryPath() {
    return binary_path;
}

test_ui_mode MainGetTestUIMode() {
    return test_ui_mode_cli;
}

void MainToggleConsoleWindow() {
}
