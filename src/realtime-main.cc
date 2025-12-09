// realtime-main.cc
// Real-time audio denoising using PortAudio
// Captures audio from microphone, applies DFSMN denoising, outputs to speakers

#include <atomic>
#include <chrono>
#include <cstring>
#include <iostream>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>
#include <deque>
#include <csignal>
#include <unistd.h>
#include <sys/select.h>

#include "portaudio.h"
#include "stream-denoiser.h"

namespace {

// Global flag for graceful shutdown
std::atomic<bool> g_running{true};

void SignalHandler(int signum) {
  std::cout << "\nReceived signal " << signum << ", shutting down...\n";
  g_running = false;
}

struct AudioContext {
  dfsmn::StreamDenoiser* denoiser = nullptr;

  // Ring buffers for audio data
  std::mutex mutex;
  std::deque<float> input_queue;
  std::deque<float> output_queue;

  // Statistics
  std::atomic<int64_t> samples_in{0};
  std::atomic<int64_t> samples_out{0};
  std::atomic<int64_t> underruns{0};
  std::atomic<int64_t> overruns{0};

  // Configuration
  int32_t chunk_size = 960;  // 20ms at 48kHz
  int32_t max_queue_size = 48000;  // 1 second buffer
  bool bypass = false;  // Bypass denoising
};

// PortAudio callback for full-duplex audio
static int AudioCallback(const void* input_buffer, void* output_buffer,
                         unsigned long frames_per_buffer,
                         const PaStreamCallbackTimeInfo* time_info,
                         PaStreamCallbackFlags status_flags,
                         void* user_data) {
  (void)time_info;

  auto* ctx = static_cast<AudioContext*>(user_data);
  const float* in = static_cast<const float*>(input_buffer);
  float* out = static_cast<float*>(output_buffer);

  if (status_flags & paInputOverflow) {
    ctx->overruns++;
  }
  if (status_flags & paOutputUnderflow) {
    ctx->underruns++;
  }

  std::lock_guard<std::mutex> lock(ctx->mutex);

  // Add input samples to queue
  if (in) {
    for (unsigned long i = 0; i < frames_per_buffer; ++i) {
      if (ctx->input_queue.size() < static_cast<size_t>(ctx->max_queue_size)) {
        ctx->input_queue.push_back(in[i]);
      }
    }
    ctx->samples_in += frames_per_buffer;
  }

  // Get output samples from queue
  for (unsigned long i = 0; i < frames_per_buffer; ++i) {
    if (!ctx->output_queue.empty()) {
      out[i] = ctx->output_queue.front();
      ctx->output_queue.pop_front();
    } else {
      out[i] = 0.0f;  // Output silence if no data available
    }
  }

  return g_running ? paContinue : paComplete;
}

void PrintUsage(const char* program) {
  std::cout << "Usage: " << program << " [options]\n"
            << "\nOptions:\n"
            << "  --model=<path>     Path to ONNX model (required)\n"
            << "  --threads=<n>      Number of threads (default: 1)\n"
            << "  --input-device=<n> Input device index (default: default device)\n"
            << "  --output-device=<n> Output device index (default: default device)\n"
            << "  --list-devices     List available audio devices and exit\n"
            << "  --bypass           Start in bypass mode (no denoising)\n"
            << "  --help             Show this help\n"
            << "\nControls during operation:\n"
            << "  b                  Toggle bypass mode\n"
            << "  s                  Show statistics\n"
            << "  q                  Quit\n";
}

void ListDevices() {
  int num_devices = Pa_GetDeviceCount();
  if (num_devices < 0) {
    std::cerr << "Error getting device count: " << Pa_GetErrorText(num_devices) << "\n";
    return;
  }

  std::cout << "\nAvailable audio devices:\n";
  std::cout << "========================\n";

  int default_input = Pa_GetDefaultInputDevice();
  int default_output = Pa_GetDefaultOutputDevice();

  for (int i = 0; i < num_devices; ++i) {
    const PaDeviceInfo* info = Pa_GetDeviceInfo(i);
    if (!info) continue;

    std::cout << "[" << i << "] " << info->name;
    if (i == default_input) std::cout << " (default input)";
    if (i == default_output) std::cout << " (default output)";
    std::cout << "\n";
    std::cout << "    In: " << info->maxInputChannels
              << " ch, Out: " << info->maxOutputChannels << " ch"
              << ", Sample rate: " << info->defaultSampleRate << " Hz\n";
  }
}

void PrintStatistics(const AudioContext& ctx) {
  std::cout << "\n=== Statistics ===\n";
  std::cout << "Samples in:     " << ctx.samples_in << "\n";
  std::cout << "Samples out:    " << ctx.samples_out << "\n";
  std::cout << "Input queue:    " << ctx.input_queue.size() << " samples\n";
  std::cout << "Output queue:   " << ctx.output_queue.size() << " samples\n";
  std::cout << "Underruns:      " << ctx.underruns << "\n";
  std::cout << "Overruns:       " << ctx.overruns << "\n";
  std::cout << "Bypass mode:    " << (ctx.bypass ? "ON" : "OFF") << "\n";
  std::cout << "==================\n";
}

}  // namespace

int main(int argc, char* argv[]) {
  // Parse arguments
  std::string model_path;
  int num_threads = 1;
  int input_device = -1;  // -1 means default
  int output_device = -1;
  bool list_devices = false;
  bool bypass = false;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg.find("--model=") == 0) {
      model_path = arg.substr(8);
    } else if (arg.find("--threads=") == 0) {
      num_threads = std::stoi(arg.substr(10));
    } else if (arg.find("--input-device=") == 0) {
      input_device = std::stoi(arg.substr(15));
    } else if (arg.find("--output-device=") == 0) {
      output_device = std::stoi(arg.substr(16));
    } else if (arg == "--list-devices") {
      list_devices = true;
    } else if (arg == "--bypass") {
      bypass = true;
    } else if (arg == "--help" || arg == "-h") {
      PrintUsage(argv[0]);
      return 0;
    } else {
      std::cerr << "Unknown option: " << arg << "\n";
      PrintUsage(argv[0]);
      return 1;
    }
  }

  // Initialize PortAudio
  PaError err = Pa_Initialize();
  if (err != paNoError) {
    std::cerr << "PortAudio initialization failed: " << Pa_GetErrorText(err) << "\n";
    return 1;
  }

  if (list_devices) {
    ListDevices();
    Pa_Terminate();
    return 0;
  }

  if (model_path.empty()) {
    std::cerr << "Error: --model is required\n";
    PrintUsage(argv[0]);
    Pa_Terminate();
    return 1;
  }

  // Setup signal handler
  signal(SIGINT, SignalHandler);
  signal(SIGTERM, SignalHandler);

  // Configure denoiser
  dfsmn::StreamDenoiserConfig config;
  config.model_path = model_path;
  config.num_threads = num_threads;
  config.sample_rate = 48000;

  std::cout << "Initializing DFSMN denoiser...\n";
  std::cout << config.ToString() << "\n";

  dfsmn::StreamDenoiser denoiser(config);
  if (!denoiser.Init()) {
    std::cerr << "Failed to initialize denoiser\n";
    Pa_Terminate();
    return 1;
  }

  std::cout << "Denoiser initialized:\n";
  std::cout << "  Latency: " << denoiser.GetLatencyMs() << " ms\n";
  std::cout << "  Chunk size: " << denoiser.GetChunkSizeSamples() << " samples\n";

  // Setup audio context
  AudioContext ctx;
  ctx.denoiser = &denoiser;
  ctx.chunk_size = denoiser.GetChunkSizeSamples();
  ctx.bypass = bypass;

  // Setup audio stream parameters
  PaStreamParameters input_params;
  input_params.device = (input_device >= 0) ? input_device : Pa_GetDefaultInputDevice();
  input_params.channelCount = 1;  // Mono
  input_params.sampleFormat = paFloat32;
  input_params.suggestedLatency = Pa_GetDeviceInfo(input_params.device)->defaultLowInputLatency;
  input_params.hostApiSpecificStreamInfo = nullptr;

  PaStreamParameters output_params;
  output_params.device = (output_device >= 0) ? output_device : Pa_GetDefaultOutputDevice();
  output_params.channelCount = 1;  // Mono
  output_params.sampleFormat = paFloat32;
  output_params.suggestedLatency = Pa_GetDeviceInfo(output_params.device)->defaultLowOutputLatency;
  output_params.hostApiSpecificStreamInfo = nullptr;

  std::cout << "\nUsing input device: " << Pa_GetDeviceInfo(input_params.device)->name << "\n";
  std::cout << "Using output device: " << Pa_GetDeviceInfo(output_params.device)->name << "\n";

  // Open audio stream
  PaStream* stream = nullptr;
  int frames_per_buffer = ctx.chunk_size;  // 20ms chunks

  err = Pa_OpenStream(&stream,
                      &input_params,
                      &output_params,
                      config.sample_rate,
                      frames_per_buffer,
                      paClipOff,
                      AudioCallback,
                      &ctx);

  if (err != paNoError) {
    std::cerr << "Failed to open audio stream: " << Pa_GetErrorText(err) << "\n";
    Pa_Terminate();
    return 1;
  }

  // Start audio stream
  err = Pa_StartStream(stream);
  if (err != paNoError) {
    std::cerr << "Failed to start audio stream: " << Pa_GetErrorText(err) << "\n";
    Pa_CloseStream(stream);
    Pa_Terminate();
    return 1;
  }

  std::cout << "\n*** Real-time denoising started ***\n";
  std::cout << "Press 'b' to toggle bypass, 's' for stats, 'q' to quit\n\n";

  // Processing loop
  std::vector<float> process_buffer(ctx.chunk_size * 4);
  std::vector<float> output_buffer(ctx.chunk_size * 4);

  while (g_running) {
    // Check for keyboard input (non-blocking)
    fd_set fds;
    struct timeval tv;
    FD_ZERO(&fds);
    FD_SET(0, &fds);
    tv.tv_sec = 0;
    tv.tv_usec = 10000;  // 10ms timeout

    if (select(1, &fds, nullptr, nullptr, &tv) > 0) {
      char c;
      if (read(0, &c, 1) > 0) {
        switch (c) {
          case 'b':
          case 'B':
            ctx.bypass = !ctx.bypass;
            std::cout << "Bypass mode: " << (ctx.bypass ? "ON" : "OFF") << "\n";
            break;
          case 's':
          case 'S':
            PrintStatistics(ctx);
            break;
          case 'q':
          case 'Q':
            g_running = false;
            break;
        }
      }
    }

    // Process audio from input queue
    int32_t available;
    {
      std::lock_guard<std::mutex> lock(ctx.mutex);
      available = ctx.input_queue.size();
    }

    if (available >= ctx.chunk_size) {
      // Get samples from input queue
      {
        std::lock_guard<std::mutex> lock(ctx.mutex);
        for (int i = 0; i < ctx.chunk_size; ++i) {
          process_buffer[i] = ctx.input_queue.front();
          ctx.input_queue.pop_front();
        }
      }

      int32_t output_count;
      if (ctx.bypass) {
        // Bypass mode: pass through without processing
        std::memcpy(output_buffer.data(), process_buffer.data(),
                    ctx.chunk_size * sizeof(float));
        output_count = ctx.chunk_size;
      } else {
        // Apply denoising
        output_count = denoiser.Process(process_buffer.data(), ctx.chunk_size,
                                        output_buffer.data());
      }

      // Add to output queue
      if (output_count > 0) {
        std::lock_guard<std::mutex> lock(ctx.mutex);
        for (int32_t i = 0; i < output_count; ++i) {
          if (ctx.output_queue.size() < static_cast<size_t>(ctx.max_queue_size)) {
            ctx.output_queue.push_back(output_buffer[i]);
          }
        }
        ctx.samples_out += output_count;
      }
    } else {
      // No data to process, sleep briefly
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  }

  // Cleanup
  std::cout << "\nStopping audio stream...\n";
  Pa_StopStream(stream);
  Pa_CloseStream(stream);
  Pa_Terminate();

  PrintStatistics(ctx);
  std::cout << "Done.\n";

  return 0;
}
