// dfsmn-denoiser.h
// DFSMN ANS (Acoustic Noise Suppression) Denoiser
// Based on ModelScope speech_dfsmn_ans_psm_48k_causal model

#ifndef DFSMN_DENOISER_H_
#define DFSMN_DENOISER_H_

#include <memory>
#include <string>
#include <vector>

namespace dfsmn {

struct DenoiserConfig {
  std::string model_path;     // Path to ONNX model
  int32_t num_threads = 1;    // Number of threads for ORT

  // Model parameters (from ModelScope pipeline)
  int32_t sample_rate = 48000;
  int32_t n_fft = 1920;           // 40ms at 48kHz
  int32_t hop_length = 960;       // 20ms at 48kHz
  int32_t win_length = 1920;
  int32_t num_mel_bins = 120;     // Fbank dimension
  float frame_length_ms = 40.0f;
  float frame_shift_ms = 20.0f;
  float dither = 1.0f;

  std::string ToString() const;
};

struct DenoisedAudio {
  std::vector<float> samples;   // Denoised audio samples (normalized to [-1, 1])
  int32_t sample_rate;
};

class DfsmnDenoiser {
 public:
  explicit DfsmnDenoiser(const DenoiserConfig& config);
  ~DfsmnDenoiser();

  // Disable copy
  DfsmnDenoiser(const DfsmnDenoiser&) = delete;
  DfsmnDenoiser& operator=(const DfsmnDenoiser&) = delete;

  /**
   * Run denoising on audio samples
   * @param samples Audio samples normalized to [-1, 1]
   * @param n Number of samples
   * @param sample_rate Sample rate of input audio
   * @return Denoised audio
   */
  DenoisedAudio Run(const float* samples, int32_t n, int32_t sample_rate) const;

  int32_t GetSampleRate() const { return config_.sample_rate; }

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
  DenoiserConfig config_;
};

}  // namespace dfsmn

#endif  // DFSMN_DENOISER_H_
