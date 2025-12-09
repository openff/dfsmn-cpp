// main.cc
// DFSMN Denoiser Test Program

#include <chrono>
#include <iostream>
#include <string>

#include "dfsmn-denoiser.h"
#include "wave-reader.h"
#include "wave-writer.h"
/**
 /data/8T/modle/dfsmn/linux-dfsmn-C++/build/dfsmn-denoiser \
     --model=/data/8T/modle/dfsmn/onnx/model.onnx \
     --input=/data/8T/modle/dfsmn/speech_with_noise_48k.wav \
     --output=/tmp/denoised_output_v2.wav \
     --threads=1
*/
void PrintUsage(const char* prog) {
  std::cerr << "Usage: " << prog << " [options]\n\n"
            << "Options:\n"
            << "  --model=<path>      Path to ONNX model (required)\n"
            << "  --input=<path>      Path to input WAV file (required)\n"
            << "  --output=<path>     Path to output WAV file (required)\n"
            << "  --threads=<num>     Number of threads (default: 1)\n"
            << "  --help              Show this help message\n\n"
            << "Example:\n"
            << "  " << prog << " --model=model.onnx --input=noisy.wav --output=clean.wav\n";
}

int main(int argc, char* argv[]) {
  std::string model_path;
  std::string input_path;
  std::string output_path;
  int32_t num_threads = 1;

  // Parse arguments
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg.find("--model=") == 0) {
      model_path = arg.substr(8);
    } else if (arg.find("--input=") == 0) {
      input_path = arg.substr(8);
    } else if (arg.find("--output=") == 0) {
      output_path = arg.substr(9);
    } else if (arg.find("--threads=") == 0) {
      num_threads = std::stoi(arg.substr(10));
    } else if (arg == "--help" || arg == "-h") {
      PrintUsage(argv[0]);
      return 0;
    } else {
      std::cerr << "Unknown argument: " << arg << "\n";
      PrintUsage(argv[0]);
      return 1;
    }
  }

  if (model_path.empty() || input_path.empty() || output_path.empty()) {
    std::cerr << "Error: --model, --input, and --output are required\n\n";
    PrintUsage(argv[0]);
    return 1;
  }

  // Read input audio
  std::cerr << "Reading input: " << input_path << "\n";
  int32_t sample_rate = 0;
  bool is_ok = false;
  std::vector<float> samples = sherpa_onnx::ReadWave(input_path, &sample_rate, &is_ok);

  if (!is_ok) {
    std::cerr << "Error: Failed to read " << input_path << "\n";
    return 1;
  }

  float duration = static_cast<float>(samples.size()) / sample_rate;
  std::cerr << "  Sample rate: " << sample_rate << " Hz\n";
  std::cerr << "  Duration: " << duration << " seconds\n";
  std::cerr << "  Samples: " << samples.size() << "\n";

  // Create denoiser
  dfsmn::DenoiserConfig config;
  config.model_path = model_path;
  config.num_threads = num_threads;

  std::cerr << "\n" << config.ToString() << "\n";

  dfsmn::DfsmnDenoiser denoiser(config);

  // Run denoising
  std::cerr << "Running denoising...\n";
  auto start = std::chrono::steady_clock::now();

  dfsmn::DenoisedAudio result = denoiser.Run(samples.data(), samples.size(), sample_rate);

  auto end = std::chrono::steady_clock::now();
  float elapsed = std::chrono::duration<float>(end - start).count();

  std::cerr << "Done!\n";
  std::cerr << "  Processing time: " << elapsed << " seconds\n";
  std::cerr << "  RTF (Real-Time Factor): " << elapsed / duration << "\n";

  // Write output
  std::cerr << "\nWriting output: " << output_path << "\n";

  // WriteWave expects float samples in [-1, 1] range
  is_ok = sherpa_onnx::WriteWave(output_path, result.sample_rate,
                                  result.samples.data(), result.samples.size());

  if (!is_ok) {
    std::cerr << "Error: Failed to write " << output_path << "\n";
    return 1;
  }

  std::cerr << "Success! Denoised audio saved to: " << output_path << "\n";
  return 0;
}
