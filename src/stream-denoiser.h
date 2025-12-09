// stream-denoiser.h
// Real-time streaming DFSMN denoiser
// Supports chunk-based processing for low-latency audio denoising

#ifndef STREAM_DENOISER_H_
#define STREAM_DENOISER_H_

#include <memory>
#include <string>
#include <vector>
#include <functional>
#include <atomic>

namespace dfsmn {

struct StreamDenoiserConfig {
  std::string model_path;           // Path to ONNX model
  int32_t num_threads = 1;          // Number of threads for ORT
  int32_t sample_rate = 48000;      // Must be 48kHz for this model
  int32_t chunk_size_ms = 20;       // Processing chunk size in ms (should be multiple of hop_length)
  int32_t lookahead_ms = 0;         // Future context for better quality (adds latency)

  // Model parameters (fixed for DFSMN model)
  int32_t n_fft = 1920;             // 40ms at 48kHz
  int32_t hop_length = 960;         // 20ms at 48kHz
  int32_t win_length = 1920;
  int32_t num_mel_bins = 120;       // Fbank dimension
  float frame_length_ms = 40.0f;
  float frame_shift_ms = 20.0f;
  float dither = 0.0f;              // Disable dither for streaming (reduces noise)

  std::string ToString() const;
};

// Callback function type for processed audio output
// samples: denoised audio samples (normalized to [-1, 1])
// n: number of samples
using AudioOutputCallback = std::function<void(const float* samples, int32_t n)>;

class StreamDenoiser {
 public:
  explicit StreamDenoiser(const StreamDenoiserConfig& config);
  ~StreamDenoiser();

  // Disable copy
  StreamDenoiser(const StreamDenoiser&) = delete;
  StreamDenoiser& operator=(const StreamDenoiser&) = delete;

  // Enable move
  StreamDenoiser(StreamDenoiser&&) noexcept;
  StreamDenoiser& operator=(StreamDenoiser&&) noexcept;

  /**
   * Initialize the stream denoiser
   * Must be called before processing audio
   * @return true if initialization succeeded
   */
  bool Init();

  /**
   * Reset the internal state (clear buffers)
   * Call this when starting a new audio stream
   */
  void Reset();

  /**
   * Process a chunk of audio samples
   * @param samples Input audio samples (normalized to [-1, 1])
   * @param n Number of input samples
   * @param output Output buffer for denoised samples (must have at least n elements)
   * @return Number of output samples produced (may be less than n due to buffering)
   */
  int32_t Process(const float* samples, int32_t n, float* output);

  /**
   * Process with callback - for real-time streaming
   * @param samples Input audio samples (normalized to [-1, 1])
   * @param n Number of input samples
   * @param callback Callback function to receive denoised audio
   */
  void ProcessWithCallback(const float* samples, int32_t n,
                           const AudioOutputCallback& callback);

  /**
   * Flush remaining audio in the buffer
   * Call this at the end of the stream to get any remaining samples
   * @param output Output buffer
   * @param max_samples Maximum number of samples to output
   * @return Number of output samples produced
   */
  int32_t Flush(float* output, int32_t max_samples);

  /**
   * Get the processing latency in samples
   * @return Latency in samples
   */
  int32_t GetLatencySamples() const;

  /**
   * Get the processing latency in milliseconds
   * @return Latency in milliseconds
   */
  float GetLatencyMs() const;

  /**
   * Get the expected input chunk size
   * @return Recommended input chunk size in samples
   */
  int32_t GetChunkSizeSamples() const;

  /**
   * Get sample rate
   * @return Sample rate in Hz
   */
  int32_t GetSampleRate() const { return config_.sample_rate; }

  /**
   * Check if initialized
   * @return true if initialized
   */
  bool IsInitialized() const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
  StreamDenoiserConfig config_;
};

}  // namespace dfsmn

#endif  // STREAM_DENOISER_H_
