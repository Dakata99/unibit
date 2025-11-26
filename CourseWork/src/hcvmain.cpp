#include <map>
#include <string>
#include <vector>
#include <memory>
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>

// Third party includes
#include <args.hxx>
#include <spdlog/spdlog.h>
#include <spdlog/fmt/ranges.h>

// Tenforflow Lite includes
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/version.h"
#include "tensorflow/lite/error_reporter.h"

struct BioMetric {
    std::string shortname;
    float value;
    std::string name;
    std::string description;

    BioMetric(const std::string& _shortname, float _value = 0.0, const std::string& _name = "", const std::string& _description = "")
        : shortname(_shortname), value(_value), name(_name), description(_description) {}
};

std::vector<BioMetric> biometrics = {
    BioMetric("AST"),
    BioMetric("CHE"),
    BioMetric("ALT"),
    BioMetric("ALP"),
    BioMetric("GGT"),
};

std::vector<std::pair<std::string, std::string>> conditions = {
    { "0", "Blood donor" },
    { "0s", "Suspected blood donor" },
    { "1", "Hepatitis" },
    { "2", "Fibrosis" },
    { "3", "Cirrhosis" },
};

void printTensorInfo(const TfLiteTensor* t);
void printModelInfo(const tflite::Interpreter& interpreter);

int main(int argc, char* argv[]) {
    spdlog::set_pattern("[%^%l%$] %v");

    // TODO: description
    args::ArgumentParser parser("HCV - TODO: description");
    args::HelpFlag help(parser, "help", "Display help", {'h', "help"});

    args::Group logging(parser, "Logging options", args::Group::Validators::DontCare);
    args::Flag verbose(logging, "verbose", "Enable verbose output", {'v', "verbose"});

    args::Group input(parser, "Input options", args::Group::Validators::DontCare);
    args::Positional<std::string> nnmodel(input, "model", "TFLite model to load");

    // TODO: make these dynamic, based on the metrics???
    args::ValueFlag<float> metric1(input, biometrics[0].shortname, biometrics[0].description, {biometrics[0].shortname});
    args::ValueFlag<float> metric2(input, biometrics[1].shortname, biometrics[1].description, {biometrics[1].shortname});
    args::ValueFlag<float> metric3(input, biometrics[2].shortname, biometrics[2].description, {biometrics[2].shortname});
    args::ValueFlag<float> metric4(input, biometrics[3].shortname, biometrics[3].description, {biometrics[3].shortname});
    args::ValueFlag<float> metric5(input, biometrics[4].shortname, biometrics[4].description, {biometrics[4].shortname});

    try {
        parser.ParseCLI(argc, argv);
    } catch (args::Help) {
        std::ostringstream oss;
        oss << parser;
        spdlog::info("{}", oss.str());
        return 0;
    } catch (args::ParseError e) {
        std::ostringstream oss;
        oss << parser;
        spdlog::error("{}", e.what());
        spdlog::error("{}", oss.str());
        return 1;
    }

    if (verbose) {
        spdlog::info("Verbose mode enabled");
        spdlog::set_level(spdlog::level::debug);
    }
    spdlog::debug("Input file: {}", args::get(nnmodel));

    // Set up input arguments
    biometrics[0].value = args::get(metric1);
    biometrics[1].value = args::get(metric2);
    biometrics[2].value = args::get(metric3);
    biometrics[3].value = args::get(metric4);
    biometrics[4].value = args::get(metric5);

    for (const BioMetric& bm : biometrics) {
        spdlog::debug("Metric: {}, value: {}", bm.shortname, bm.value);
    }

    // TFLite version check
    spdlog::info("TFLite version: {}", TFLITE_VERSION_STRING);

    // FIXME: how to use this???
    tflite::StderrReporter reporter;

    // Load model
    std::unique_ptr<tflite::impl::FlatBufferModel> model = tflite::FlatBufferModel::BuildFromFile(args::get(nnmodel).c_str(), &reporter);

    if (!model) {
        spdlog::error("Failed to load model from: {}", args::get(nnmodel));
        return 1;
    }

    // Create interpreter with built-in ops
    tflite::ops::builtin::BuiltinOpResolver resolver;
    std::unique_ptr<tflite::Interpreter> interpreter;

    tflite::InterpreterBuilder(*model, resolver)(&interpreter);
    if (!interpreter) {
        spdlog::error("Failed to create interpreter");
        return 1;
    }

    // Allocate tensors
    if (interpreter->AllocateTensors() != kTfLiteOk) {
        spdlog::error("Failed to allocate tensors");
        return 1;
    }

    // Print some information about the model
    printModelInfo(*interpreter);

    int input_idx = interpreter->inputs()[0];
    TfLiteTensor* input_tensor = interpreter->tensor(input_idx);

    // Get input tensor
    float* input_data = interpreter->typed_tensor<float>(input_idx);
    for (int i = 0; i < input_tensor->bytes / sizeof(float); i++) {
        input_data[i] = biometrics[i].value;
    }

    // Run inference
    if (interpreter->Invoke() != kTfLiteOk) {
        spdlog::error("Failed to invoke TFLite interpreter");
        return 1;
    }

    // Get output tensor
    int output_idx = interpreter->outputs()[0];
    TfLiteTensor* output_tensor = interpreter->tensor(output_idx);

    float* output_data = interpreter->typed_tensor<float>(output_idx);
    for (int i = 0; i < output_tensor->bytes / sizeof(float); i++) {
        spdlog::debug("Condition[{}] = {}", i, output_data[i]);
    }

    // Evaluate results
    // TODO: Currently, the evaluation takes the highest probability, 
    //       but what decision should be mabe if probabilities for 
    //       all conditions are pretty similar???
    auto it = std::max_element(output_data, output_data + biometrics.size());
    size_t idx = std::distance(output_data, it);

    spdlog::info("Condition: {}, value: {}", conditions.at(idx).second, *it);

    return 0;
}

void printTensorInfo(const TfLiteTensor* t) {
    spdlog::debug("Name: {}", (t->name ? t->name : "<none>"));
    if (t->dims) {
        spdlog::debug("Dims: [{}]", fmt::join(t->dims->data, t->dims->data + t->dims->size, ", "));
    }
    switch (t->type) {
        case kTfLiteFloat32:
            spdlog::debug("Type: FLOAT32");
            break;
        default:
            spdlog::debug("Type: UNKNOWN");
            break;
    }
}

void printModelInfo(const tflite::Interpreter& interpreter) {
    spdlog::debug("================== Model summary ==================");
    spdlog::debug("# of tensors: {}", interpreter.tensors_size());
    spdlog::debug("# of ops: {}", interpreter.nodes_size());

    spdlog::debug("========= Inputs =========");
    for (int idx : interpreter.inputs()) {
        const TfLiteTensor* t = interpreter.tensor(idx);
        printTensorInfo(t);
    }

    spdlog::debug("========= Outputs =========");
    for (int idx : interpreter.outputs()) {
        const TfLiteTensor* t = interpreter.tensor(idx);
        printTensorInfo(t);
    }
    spdlog::debug("===================================================");
}
